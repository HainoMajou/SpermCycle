import os
import time
import torch
import torch.distributed as dist
from collections import OrderedDict
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer


def parse_gpu_ids():
    """Parse gpu_ids from command line before full option parsing."""
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gpu_ids', type=str, default='0,1')
    args, _ = parser.parse_known_args()
    gpu_ids = [int(x) for x in args.gpu_ids.split(',') if x.strip()]
    return gpu_ids


def setup_ddp(gpu_ids=None):
    """Initialize DDP environment. If gpu_ids is provided, local_rank will be mapped to these IDs."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # 如果指定了 gpu_ids，将 local_rank 映射到对应的 GPU
        if gpu_ids is not None and len(gpu_ids) > 0:
            if local_rank < len(gpu_ids):
                actual_gpu = gpu_ids[local_rank]
            else:
                raise ValueError(f"local_rank {local_rank} >= len(gpu_ids) {len(gpu_ids)}. "
                               f"Make sure --nproc_per_node <= len(gpu_ids)")
        else:
            actual_gpu = local_rank
        
        # 设置当前进程使用的 GPU
        torch.cuda.set_device(actual_gpu)
        
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        return rank, world_size, local_rank, actual_gpu
    else:
        # Single GPU fallback
        if gpu_ids is not None and len(gpu_ids) > 0:
            actual_gpu = gpu_ids[0]
            torch.cuda.set_device(actual_gpu)
        else:
            actual_gpu = 0
        return 0, 1, 0, actual_gpu


def cleanup_ddp():
    """Clean up DDP resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if current process is the main process (rank 0)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def reduce_losses(losses, world_size, device):
    """Reduce losses across all processes and return the average.
    Args:
        losses (OrderedDict): Dictionary of loss names to values
        world_size (int): Number of processes
        device: The device to use for tensor operations
    Returns:
        OrderedDict: Averaged losses across all processes
    """
    if not dist.is_initialized() or world_size == 1:
        return losses
    
    # Convert losses to tensor for all_reduce - use specific device
    loss_names = list(losses.keys())
    loss_values = torch.tensor([losses[name] for name in loss_names], 
                               dtype=torch.float32, device=device)
    
    # Sum across all processes
    dist.all_reduce(loss_values, op=dist.ReduceOp.SUM)
    
    # Average
    loss_values = loss_values / world_size
    
    # Convert back to OrderedDict
    reduced_losses = OrderedDict()
    for i, name in enumerate(loss_names):
        reduced_losses[name] = float(loss_values[i].item())
    
    return reduced_losses


if __name__ == '__main__':
    # 先解析 gpu_ids 参数
    gpu_ids = parse_gpu_ids()
    
    # Setup DDP，传入 gpu_ids 进行映射
    rank, world_size, local_rank, actual_gpu = setup_ddp(gpu_ids)
    distributed = world_size > 1
    
    try:
        # Inject DDP info into environment for options parsing
        import builtins
        _original_print = builtins.print
        
        # Suppress prints from non-main processes during initialization
        if rank != 0:
            builtins.print = lambda *args, **kwargs: None
        
        opt = TrainOptions().parse()   # get training options
        
        # Restore print
        builtins.print = _original_print
        
        # Set DDP-related options BEFORE creating dataset/model
        opt.rank = rank
        opt.world_size = world_size
        opt.local_rank = local_rank
        opt.distributed = distributed
        
        # 设置当前进程使用的 GPU（已映射到用户指定的 gpu_ids）
        opt.gpu_ids = [actual_gpu]
        
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)    # get the number of images in the dataset.
        
        if is_main_process():
            if distributed:
                print('DDP enabled: world_size=%d, using DistributedSampler' % world_size)
                print('The number of training images = %d (total), %d per GPU' % (dataset_size, dataset_size // world_size))
            else:
                print('The number of training images = %d' % dataset_size)

        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        
        # Only create visualizer on main process
        if is_main_process():
            visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
        
        total_iters = 0                # the total number of training iterations

        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
            
            if is_main_process():
                visualizer.reset()              # reset the visualizer
            
            # Set epoch for DistributedSampler to shuffle data differently each epoch
            if opt.distributed and hasattr(dataset, 'set_epoch'):
                dataset.set_epoch(epoch)
            
            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += opt.batch_size * opt.world_size  # Account for all GPUs
                epoch_iter += opt.batch_size * opt.world_size
                model.set_input(data)         # unpack data from dataset and apply preprocessing
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

                # Visualize on main process only
                if is_main_process():
                    if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                        save_result = total_iters % opt.update_html_freq == 0
                        visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                # Print losses - all processes participate in reduction, only rank 0 prints
                if total_iters % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    # Reduce losses across all GPUs to get average
                    if opt.distributed:
                        device = torch.device('cuda', actual_gpu)
                        losses = reduce_losses(losses, opt.world_size, device)
                    if is_main_process():
                        t_comp = (time.time() - iter_start_time) / opt.batch_size
                        visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

                # Save model - must synchronize all processes first
                if total_iters % opt.save_latest_freq == 0:
                    if opt.distributed:
                        dist.barrier()  # Wait for all processes before saving
                    if is_main_process():
                        print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                        save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                        model.save_networks(save_suffix)
                    if opt.distributed:
                        dist.barrier()  # Wait for save to complete before continuing

                iter_data_time = time.time()
            
            # Save model at epoch end - synchronize all processes
            if epoch % opt.save_epoch_freq == 0:
                if opt.distributed:
                    dist.barrier()  # Wait for all processes
                if is_main_process():
                    print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                    model.save_networks('latest')
                    model.save_networks(epoch)
                if opt.distributed:
                    dist.barrier()  # Wait for save to complete
                
            model.update_learning_rate()    # update learning rates in the beginning of every epoch.
            
            if is_main_process():
                print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    
    finally:
        cleanup_ddp()
