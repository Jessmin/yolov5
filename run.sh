nohup python train.py --weights /home/hjj/Desktop/weights/yolov5n.pt --cfg models/yolov5n.yaml --data /home/hjj/Desktop/dataset/dataset_crowdhuman/data.yaml \
    --hyp data/hyps/hyp.scratch-low.yaml --cache --name yolov5n_bassline --batch-size 64 --workers 4 --epochs 200 > logs/yolov5n_bassline.out 2>&1 & tail -f logs/yolov5n_bassline.out

nohup python compress.py --model yolov5n --dataset VOC --data /home/hjj/Desktop/dataset/dataset_crowdhuman/data.yaml --batch 64 --epochs 100 --weights runs/train/yolov5n_bassline/weights/best.pt \
    --workers 4 --initial_rate 0.06 --initial_thres 6. --topk 0.8 --exp --cache --name yolov5n_prune > logs/yolov5n_prune.out 2>&1 & tail -f logs/yolov5n_prune.out

nohup python compress.py --model yolov5n --dataset VOC --data /home/hjj/Desktop/dataset/dataset_crowdhuman/data.yaml --batch 64 --epochs 100 --weights runs/train/yolov5n_bassline/weights/best.pt \
    --workers 4 --initial_rate 0.06 --initial_thres 6. --topk 0.8 --exp --cache --name  yolov5n_prune2 --prunemethod L2 > logs/yolov5n_prune2.out 2>&1 & tail -f logs/yolov5n_prune2.out

nohup python compress.py --model yolov5n --dataset VOC --data /home/hjj/Desktop/dataset/dataset_crowdhuman/data.yaml --batch 64 --epochs 100 --weights runs/train/yolov5n_bassline/weights/best.pt \
    --workers 4 --initial_rate 0.05 --initial_thres 5. --topk 0.8 --exp --cache --name yolov5n_prune3 > logs/yolov5n_prune3.out 2>&1 & tail -f logs/yolov5n_prune3.out

nohup python compress.py --model yolov5n --dataset VOC --data /home/hjj/Desktop/dataset/dataset_crowdhuman/data.yaml --batch 64 --epochs 100 --weights runs/train/yolov5n_bassline/weights/best.pt \
    --workers 4 --initial_rate 0.2 --initial_thres 10. --topk 0.8 --exp --cache --name yolov5n_prune4 > logs/yolov5n_prune4.out 2>&1 & tail -f logs/yolov5n_prune4.out

python val.py --data /home/hjj/Desktop/dataset/dataset_crowdhuman/data.yaml --weights runs/train/yolov5n_bassline/weights/best.pt --task test --name yolov5n_bassline --exist-ok
python val.py --data /home/hjj/Desktop/dataset/dataset_crowdhuman/data.yaml --weights runs/train/yolov5n_prune/weights/best.pt --task test --name yolov5n_prune --exist-ok
python val.py --data /home/hjj/Desktop/dataset/dataset_crowdhuman/data.yaml --weights runs/train/yolov5n_prune2/weights/best.pt --task test --name yolov5n_prune2 --exist-ok
python val.py --data /home/hjj/Desktop/dataset/dataset_crowdhuman/data.yaml --weights runs/train/yolov5n_prune3/weights/best.pt --task test --name yolov5n_prune3 --exist-ok
python val.py --data /home/hjj/Desktop/dataset/dataset_crowdhuman/data.yaml --weights runs/train/yolov5n_prune4/weights/best.pt --task test --name yolov5n_prune4 --exist-ok

python compress.py --model yolov5n --dataset VOC --data /home/hjj/Desktop/dataset/dataset_crowdhuman/data.yaml --batch 64 --epochs 100 --weights runs/train/yolov5n_bassline/weights/best.pt \
    --workers 4 --initial_rate 0.2 --initial_thres 10. --topk 0.8 --exp --cache --name yolov5n_prune_exp

nohup python train.py --weights /home/hjj/Desktop/weights/yolov5m.pt --cfg models/yolov5m.yaml --data /home/hjj/Desktop/dataset/dataset_crowdhuman/data.yaml \
    --hyp data/hyps/hyp.scratch-med.yaml --cache --name yolov5m_bassline --batch-size 32 --workers 4 --epochs 200 > logs/yolov5m_bassline.out 2>&1 & tail -f logs/yolov5m_bassline.out

nohup python compress.py --model yolov5m --dataset VOC --data /home/hjj/Desktop/dataset/dataset_crowdhuman/data.yaml --batch 32 --epochs 100 --weights runs/train/yolov5m_bassline/weights/best.pt \
    --workers 4 --initial_rate 0.2 --initial_thres 10. --topk 0.8 --exp --cache --name yolov5m_prune --hyp data/hyps/hyp.scratch-yolov5m-prune.yaml > logs/yolov5m_prune.out 2>&1 & tail -f logs/yolov5m_prune.out

nohup python compress.py --model yolov5m --dataset VOC --data /home/hjj/Desktop/dataset/dataset_crowdhuman/data.yaml --batch 32 --epochs 100 --weights runs/train/yolov5m_bassline/weights/best.pt \
    --workers 4 --initial_rate 0.2 --initial_thres 10. --topk 0.8 --exp --cache --name yolov5m_prune2 --hyp data/hyps/hyp.scratch-yolov5m-prune2.yaml > logs/yolov5m_prune2.out 2>&1 & tail -f logs/yolov5m_prune2.out

nohup python compress.py --model yolov5m --dataset VOC --data /home/hjj/Desktop/dataset/dataset_crowdhuman/data.yaml --batch 32 --epochs 100 --weights runs/train/yolov5m_bassline/weights/best.pt \
    --workers 4 --initial_rate 0.06 --initial_thres 6. --topk 0.8 --exp --cache --name yolov5m_prune3 --hyp data/hyps/hyp.scratch-yolov5m-prune2.yaml > logs/yolov5m_prune3.out 2>&1 & tail -f logs/yolov5m_prune3.out

nohup python compress.py --model yolov5m --dataset VOC --data /home/hjj/Desktop/dataset/dataset_crowdhuman/data.yaml --batch 32 --epochs 100 --weights runs/train/yolov5m_bassline/weights/best.pt \
    --workers 4 --initial_rate 0.4 --initial_thres 20. --topk 0.8 --exp --cache --name yolov5m_prune4 --hyp data/hyps/hyp.scratch-yolov5m-prune.yaml > logs/yolov5m_prune4.out 2>&1 & tail -f logs/yolov5m_prune4.out

python val.py --data /home/hjj/Desktop/dataset/dataset_crowdhuman/data.yaml --weights runs/train/yolov5m_bassline/weights/best.pt --task test --name yolov5m_bassline --exist-ok
python val.py --data /home/hjj/Desktop/dataset/dataset_crowdhuman/data.yaml --weights runs/train/yolov5m_prune/weights/best.pt --task test --name yolov5m_prune --exist-ok
python val.py --data /home/hjj/Desktop/dataset/dataset_crowdhuman/data.yaml --weights runs/train/yolov5m_prune2/weights/best.pt --task test --name yolov5m_prune2 --exist-ok
python val.py --data /home/hjj/Desktop/dataset/dataset_crowdhuman/data.yaml --weights runs/train/yolov5m_prune3/weights/best.pt --task test --name yolov5m_prune3 --exist-ok
python val.py --data /home/hjj/Desktop/dataset/dataset_crowdhuman/data.yaml --weights runs/train/yolov5m_prune4/weights/best.pt --task test --name yolov5m_prune4 --exist-ok

torch_pruning==0.2.7