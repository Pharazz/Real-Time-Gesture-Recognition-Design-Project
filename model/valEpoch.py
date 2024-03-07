import torch
import time
import sys
import torch
import torch.distributed as dist
#from utils import AverageMeter, calculate_accuracy
def val_epoch(epoch,
 typeValid,
 data_loader,
 model,
 criterion,
 device,
 logger,
 tb_writer=None,
 distributed=False):
 print('validation at epoch {}'.format(epoch))
 model.eval()
 batch_time = AverageMeter()
 data_time = AverageMeter()
 losses = AverageMeter()
 accuracies = AverageMeter()
 val_batch_time = AverageMeter()
 val_data_time = AverageMeter()
 val_losses = AverageMeter()
 val_accuracies = AverageMeter()
 end_time = time.time()
 val_end_time = time.time()
 for i_epoch in range(0,epoch):
 with torch.no_grad():
 for i, (inputs, targets) in enumerate(data_loader):
 data_time.update(time.time() - end_time)
 targets = targets.to(device, non_blocking=True)
 outputs = model(inputs)
 loss = criterion(outputs, targets)
 acc = calculate_accuracy(outputs, targets)

 val_losses.update(loss.item(), inputs.size(0))
 val_accuracies.update(acc, inputs.size(0))

 val_batch_time.update(time.time() - val_end_time)
 val_end_time = time.time()
 losses.update(loss.item(), inputs.size(0))
 accuracies.update(acc, inputs.size(0))
 batch_time.update(time.time() - end_time)
 end_time = time.time()
 print(typeValid+' Epoch: [{0}][{1}/{2}]\t'
 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
 'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
 i_epoch,
 i + 1,
 len(data_loader),
 batch_time=batch_time,
 data_time=data_time,
 loss=losses,
 acc=accuracies))
 if logger is not None:
 _, pred = outputs.topk(1, 1, largest=True, sorted=True)
 _, label = targets.topk(1, 1, largest=True, sorted=True)
 logger.log({ 'epoch': i_epoch,
 'batch': i,
 'loss': val_losses.avg,
 'acc': val_accuracies.avg,
 'Guess1': pred[0].item() if pred.size(0) >= 1 else 'N/A',
 'Guess2': pred[1].item() if pred.size(0) >= 2 else 'N/A',
 'Guess3': pred[2].item() if pred.size(0) >= 3 else 'N/A',
 'Guess4': pred[3].item() if pred.size(0) >= 4 else 'N/A',
 'Label1': label[0].item() if label.size(0) >= 1 else 'N/A',
 'Label2': label[1].item() if label.size(0) >= 2 else 'N/A',
 'Label3': label[2].item() if label.size(0) >= 3 else 'N/A',
 'Label4': label[3].item() if label.size(0) >= 4 else 'N/A'})
 if distributed:
 loss_sum = torch.tensor([losses.sum],
 dtype=torch.float32,
 device=device)
 loss_count = torch.tensor([losses.count],
 dtype=torch.float32,
 device=device)
 acc_sum = torch.tensor([accuracies.sum],
 dtype=torch.float32,
 device=device)
 acc_count = torch.tensor([accuracies.count],
 dtype=torch.float32,
 device=device)
 dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
 dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
 dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
 dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)
 losses.avg = loss_sum.item() / loss_count.item()
 accuracies.avg = acc_sum.item() / acc_count.item()

 if tb_writer is not None:
 tb_writer.add_scalar('val/loss', losses.avg, epoch)
 tb_writer.add_scalar('val/acc', accuracies.avg, epoch)
 return losses.avg
