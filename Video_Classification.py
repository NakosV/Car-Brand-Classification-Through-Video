import os
import cv2
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from collections import defaultdict
from PIL import Image
from ultralytics import YOLO
import subprocess, re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = '/your/own/path'
video_dir = '/your/own/path/of/the/video'
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_resnet50.pth')

# All the Hyperparameters
batch_size = 32
number_of_workers = 6
epochs_frozen = 10
epochs_unfrozen = 30
learning_rate_from_frozen = 0.001
learning_rate_from_unfrozen = 0.0001
checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
checkpoint_times = 5
yolo_confidence = 0.4
resnet_confidence = 0.4
max_missed_frames = 30
max_centroid_distance = 200
count_line_posiiton = 0.85
classify_line_position = 0.6
skip_frames = 2
block_number = 2

# Some picture augmentation to make the model better
transformation = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness = 0.3, contrast = 0.3, saturation = 0.2, hue = 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    }

def classes():
    return datasets.ImageFolder(os.path.join(data_dir, 'train')).classes

def loading_resnet(num_of_classes: int):
    model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_of_classes)    
    return model

# Freezes all the layers other than the fully connecte
def freeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = ('fc' in name)
        
# Unfreezes the last two blocks
def unfreeze(model, number_of_blocks = block_number):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    block_names = [f'layer{4 - i}' for i in range(number_of_blocks)]
    for name, module in model.named_modules():
        if any(name.startswith(b) for b in block_names):
            for param in module.parameters():
                param.requires_grad = True
    
def from_crop_to_tensor(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return transformation['test'](Image.fromarray(rgb)).unsqueeze(0)

def checkpoints(model, optimizer, scheduler, phase, epoch, best_accuracy):
    os.makedirs(checkpoint_dir, exist_ok = True)
    checkpoint = {'phase': phase, 'epoch': epoch, 'best_accuracy': best_accuracy, 'model_state': model.state_dict(), 
                  'optimizer_state': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
    path = os.path.join(checkpoint_dir, f'checkpoint_{phase}_epoch{epoch+1:03d}.pth')
    torch.save(checkpoint, path)
    
def list_checkpoints():
    if not os.path.exists(checkpoint_dir):
        return []
    return sorted(f for f in os.listdir(checkpoint_dir) if f.endswith('.pth'))

def get_checkpoint():
    checkpoints = list_checkpoints()
    if not checkpoints:
        return None, 0.0
    latest = os.path.join(checkpoint_dir, checkpoints[-1])
    data = torch.load(latest, map_location = 'cpu')
    print(f"[Info] Βρέθηκε ένα checkpoint: {checkpoints[-1]}" f"Φάση: {data['phase']} | Εποχή: {data['epoch']+1} | Accuracy: {data['best_accuracy']:.4f})")
    return latest, data['best_accuracy']

# Here is where the Training starts
def run_epoch(model, dataloader, dataset_size, criterion, optimizer, phase):
    model.train() if phase == 'train' else model.eval()
    running_loss = 0.0
    running_correct = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            if phase == 'train':
                loss.backward()
                optimizer.step()
        running_loss = running_loss + loss.item() * inputs.size(0)
        running_correct = running_correct + torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / dataset_size
    epoch_accuracy = (running_correct.double() / dataset_size).item()
    return epoch_loss, epoch_accuracy

def training():
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transformation[x]) for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = batch_size, shuffle = (x == 'train'), num_workers = number_of_workers, pin_memory = True) for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    num_of_classes = len(image_datasets['train'].classes)
    print(f"\n[Training] {num_of_classes} Κλάσεις | " f"Train: {dataset_sizes['train']} | Test: {dataset_sizes['test']}")
    resume_checkpoint, best_accuracy = get_checkpoint()
    model = loading_resnet(num_of_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    freeze(model)
    optimizer1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate_from_frozen)
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size = 5, gamma = 0.5)
    
    unfreeze(model, number_of_blocks = 2)
    optimizer2 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate_from_unfrozen)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max = epochs_unfrozen)
    
    start_phase = 'frozen'
    start_epoch_1 = 0
    start_epoch_2 = 0
    
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location = device)
        model.load_state_dict(checkpoint['model_state'])
        start_phase = checkpoint['phase']
        if start_phase == 'frozen':
            optimizer1.load_state_dict(checkpoint['optimizer_state'])
            scheduler1.load_state_dict(checkpoint['scheduler_state'])
            start_epoch_1 = checkpoint['epoch'] + 1
        else:
            optimizer2.load_state_dict(checkpoint['optimizer_state'])
            scheduler2.load_state_dict(checkpoint['scheduler_state'])
            start_epoch_1 = epochs_frozen
            start_epoch_2 = checkpoint['epoch'] + 1
        print(f"[Resume] Φάση: {start_phase} | Epoch: {checkpoint['epoch'] + 1} | " f"Best Accuracy: {best_accuracy:.4f}\n")
        
    # Training with the frozen layers
    if start_epoch_1 < epochs_frozen:
        freeze(model)
        print(f"Φάση 1: ({epochs_frozen} epochs, learning rate = {learning_rate_from_frozen})")
        for epoch in range(start_epoch_1, epochs_frozen):
            print(f'\nEpoch {epoch + 1}/{epochs_frozen}')
            for phase in ['train', 'test']:
                loss, accuracy = run_epoch(model, dataloaders[phase], dataset_sizes[phase], criterion, optimizer1, phase)
                print(f'{phase.capitalize():5} Loss: {loss:.4f} Accuracy: {accuracy:.4f}')
                if phase == 'test' and accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), model_dir)
            scheduler1.step()
            if (epoch + 1) % checkpoint_times == 0:
                checkpoints(model, optimizer1, scheduler1, 'frozen', epoch, best_accuracy)
    
    # Training with the unfrozen layers    
    unfreeze(model, number_of_blocks = 2)
    print(f"Φάση 2: ({epochs_unfrozen} epochs, learning rate = {learning_rate_from_unfrozen})")
    for epoch in range(start_epoch_2, epochs_unfrozen):
        print(f'\nEpoch {epoch + 1}/{epochs_unfrozen}')
        for phase in ['train', 'test']:
            loss, accuracy = run_epoch(model, dataloaders[phase], dataset_sizes[phase], criterion, optimizer2, phase)
            print(f'{phase.capitalize():5} Loss: {loss:.4f} Accuracy: {accuracy:.4f}')
            if phase == 'test' and accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), model_dir)
        scheduler2.step()
        if (epoch + 1) % checkpoint_times == 0:
            checkpoints(model, optimizer2, scheduler2, 'unfrozen', epoch, best_accuracy) 
    print(f' [Training] Best Test Accuracy: {best_accuracy:.4f}')
    print(f"[Training] Αποθηκεύτηκε ως '{model_dir}'")

# Here is where the Inference starts
def inference():
    class_names = classes()
    number_of_classes = len(class_names)
    print(f"[ResNet] {number_of_classes} κλάσεις: {class_names[:5]}")
    resnet = loading_resnet(number_of_classes)
    resnet.load_state_dict(torch.load(model_dir, map_location = device))
    resnet = resnet.to(device)
    resnet.eval()
    yolo = YOLO('yolov8n.pt')
    vehicle_classes = {2: 'car', 3: 'motorcycle', 7: 'truck'}
    cap = cv2.VideoCapture(video_dir)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Video] {W}x{H} {fps:.1f}fps")
    output_path = video_dir.replace('.mp4', 'Detection.mp4')
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    
    count_line_posiiton_ = int(H * count_line_posiiton)
    classify_line_position_ = int(H * classify_line_position)
    tracked_cars = {}
    brand_counts = defaultdict(int)
    next_id = 0
    frame_idx = 0
    
    try:
        res = subprocess.run(['xrandr'], capture_output = True, text = True)
        match = re.search(r'current (\d+) x (\d+)', res.stdout)
        screen_W, screen_H = (int(match.group(1)), int(match.group(2))) \
        if match else (1920, 1000)
    except Exception:
        screen_W, screen_H = 1920, 1080
    print(f"[Display] {screen_W}x{screen_H}")
    print(f"[Inference] Γραμμή μέτρησης = {count_line_posiiton_} " f"({int(count_line_posiiton * 100)}%")
    
    cv2.namedWindow('Car Detection & Brand Classification', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Car Detection & Brand Classification', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    softmax = nn.Softmax(dim = 1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx = frame_idx + 1
        
        # Brand Detection
        if frame_idx % skip_frames == 0:
            results = yolo(frame, verbose = False, conf = yolo_confidence)
            current_detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in vehicle_classes:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue
                current_detections.append((x1, y1, x2, y2))
                
            # Centroid
            matched_ids = set()
            for (x1, y1, x2, y2) in current_detections:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                best_distance, best_tid = float('inf'), None
                for tid, info in tracked_cars.items():
                    tx1, ty1, tx2, ty2 = info['box']
                    tcx = (tx1 + tx2) // 2
                    tcy = (ty1 + ty2) // 2
                    distance = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
                    if distance < best_distance:
                        best_distance, best_tid = distance, tid
                        
                if best_distance <= max_centroid_distance and best_tid not in matched_ids:
                    tracked_cars[best_tid]['box'] = (x1, y1, x2, y2)
                    tracked_cars[best_tid]['last_seen'] = frame_idx
                    
                    center_y = (y1 + y2) // 2
                    near_line = center_y >= classify_line_position_
                    unresolved = tracked_cars[best_tid]['brand'] == 'Unknown'
                    if near_line and unresolved:
                        try:
                            crop = frame[y1:y2, x1:x2]
                            tensor = from_crop_to_tensor(crop).to(device)
                            with torch.no_grad():
                                probs = softmax(resnet(tensor))
                                confidence, pred = torch.max(probs, 1)
                                if confidence.item() >= resnet_confidence:
                                    tracked_cars[best_tid]['brand'] = class_names[pred.item()]
                        except Exception:
                            pass
                    matched_ids.add(best_tid)
                else:
                    brand = 'Unknown'
                    try:
                        crop = frame[y1:y2, x1:x2]
                        tensor = from_crop_to_tensor(crop).to(device)
                        with torch.no_grad():
                            probs = softmax(resnet(tensor))
                            confidence, pred = torch.max(probs, 1)
                            if confidence.item() >= resnet_confidence:
                                brand = class_names[pred.item()]
                    except Exception:
                        pass
                    tracked_cars[next_id] = {
                        'brand': brand,
                        'box': (x1, y1, x2, y2),
                        'counted': False,
                        'last_seen': frame_idx,}
                    matched_ids.add(next_id)
                    next_id = next_id + 1
            
            stale = [tid for tid, info in tracked_cars.items() if frame_idx - info['last_seen'] > max_missed_frames]
            for tid in stale:
                del tracked_cars[tid]
                
        # The car gets registered after it passes the line       
        for tid, info in tracked_cars.items():
            x1, y1, x2, y2 = info['box']
            if not info['counted'] and (y1 + y2) // 2 >= count_line_posiiton_:
                info['counted'] = True
                brand_counts[info['brand']] += 1
                print(f" {info['brand']: < 20} " f" #{brand_counts[info['brand']]} (frame {frame_idx})")
                
        cv2.line(frame, (0, count_line_posiiton_), (W, count_line_posiiton_), (0, 255, 255), 2)
        cv2.line(frame, (0, classify_line_position_), (W, classify_line_position_), (255, 100, 0), 1)
        cv2.putText(frame, "COUNT", (10, count_line_posiiton_ - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, "CLASSIFY", (10, classify_line_position_ - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
        for tid, info in tracked_cars.items():
            x1, y1, x2, y2 = info['box']
            color = (0, 200, 0) if info['counted'] else (255, 100, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{info['brand']} [#{tid}]", (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y_off = 30
        cv2.putText(frame, "=== Counts ===", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        for brand, cnt in sorted(brand_counts.items(), key = lambda x: -x[1]):
            y_off = y_off + 25
            cv2.putText(frame, f"{brand}: {cnt}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 255, 200), 2)
        writer.write(frame)
        display_frame = cv2.resize(frame, (screen_W, screen_H), interpolation = cv2.INTER_LINEAR)
        cv2.imshow('Car Detection & Brand Classification', display_frame)
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("Καταμέτρηση Αμαξιών")
    total = 0
    for brand, cnt in sorted(brand_counts.items(), key = lambda x: -x[1]):
        print(f" {brand: < 25} {cnt}")
        total = total + cnt
    print(f" {'Σύνολο': < 25} {total}")
        
if __name__ == '__main__':
    if os.path.exists(model_dir):
        inference()
    else:
        training()
        if os.path.exists(model_dir):
            inference()
