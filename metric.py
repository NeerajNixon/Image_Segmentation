def iou_score(output, target, threshold=0.5):
    # Apply sigmoid to output and detach from the computation graph
    output = output.detach()

    # Convert outputs and targets to boolean tensors
    output_bool = output > threshold
    target_bool = target > threshold

    # Perform bitwise operations on boolean tensors
    intersection = (output_bool & target_bool).float().sum((1, 2))
    union = (output_bool | target_bool).float().sum((1, 2))

    smooth = 1e-6
    iou = (intersection + smooth) / (union + smooth)  # Avoid division by zero
    return iou.mean()