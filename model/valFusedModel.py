import torch
import time
import sys
import torch
import torch.distributed as dist
#from utils import AverageMeter, calculate_accuracy
def val_epoch_fusion(epoch,
 typeValid,
 data_loader,
 data_loader2,
 model,
 model2,
 criterion,
 device,
 logger,
 tb_writer=None,
 distributed=False):
 print('validation at epoch {}'.format(epoch))
 model.eval()
 model2.eval()
 batch_time = AverageMeter()
 data_time = AverageMeter()
 losses = AverageMeter()
 accuracies = AverageMeter()
 accuracies1 = AverageMeter()
 accuracies2 = AverageMeter()
 val_batch_time = AverageMeter()
 val_data_time = AverageMeter()
 val_losses = AverageMeter()
 val_accuracies = AverageMeter()
 end_time = time.time()
 val_end_time = time.time()
 for i_epoch in range(0,epoch):
 with torch.no_grad():
 for i, ((inputs, targets),(inputs2, targets2) ) in enumerate(zip(data_loader,data_loader2)):
 data_time.update(time.time() - end_time)
 targets = targets.to(device, non_blocking=True)
 targets2 = targets.to(device, non_blocking=True)
 outputs = model(inputs)
 outputs2 = model2(inputs2)
 out = torch.mul(outputs.add(outputs2),0.5)
 #print("Target1: ",targets)
 #print("Target2: ",targets2)
 #print("Out1: ",outputs)
 #print("Out2: ",outputs2)
 #print("Fused Out: ",out)
 #print(outputs,outputs2,out)
 loss = criterion(out, targets)
 acc = calculate_accuracy(out, targets)
 acc1 = calculate_accuracy(outputs, targets)
 acc2 = calculate_accuracy(outputs2, targets)
 val_losses.update(loss.item(), inputs.size(0))
 val_accuracies.update(acc, inputs.size(0))

 val_batch_time.update(time.time() - val_end_time)
 val_end_time = time.time()
 losses.update(loss.item(), inputs.size(0))
 accuracies.update(acc, inputs.size(0))
 accuracies1.update(acc, inputs.size(0))
 accuracies2.update(acc, inputs.size(0))
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
 '''
 print(typeValid+' Epoch: [{0}][{1}/{2}]\t'
 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
 'Acc1 {acc1.val:.3f} ({acc1.avg:.3f})\t'
 'Acc2 {acc2.val:.3f} ({acc2.avg:.3f})\t'
 'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
 i_epoch,
 i + 1,
 len(data_loader),
 batch_time=batch_time,
 data_time=data_time,
 loss=losses,
 acc=accuracies,
 acc1=accuracies,
 acc2=accuracies))
 '''
 if logger is not None:
 _, pred = out.topk(1, 1, largest=True, sorted=True)
 _, pred1 = outputs.topk(1, 1, largest=True, sorted=True)
 _, pred2 = outputs2.topk(1, 1, largest=True, sorted=True)
 _, label = targets.topk(1, 1, largest=True, sorted=True)
 _, label2 = targets2.topk(1, 1, largest=True, sorted=True)
 logger.log({ 'epoch': i_epoch,
 'batch': i,
 'loss': val_losses.avg,
 'acc': val_accuracies.avg,
 'GuessRGB': pred[0].item() if pred.size(0) >= 1 else 'N/A',
 'GuessDEP': pred2[0].item() if pred.size(0) >= 1 else 'N/A',
 'GuessFinal': pred[0].item() if pred.size(0) >= 1 else 'N/A',
 'Label1': label[0].item() if label.size(0) >= 1 else 'N/A',
 'Label2': label2[1].item() if label.size(0) >= 2 else 'N/A',
 'Output RGB': outputs,
 'Output DEP': outputs2,
 'Output Fusion': out})
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
