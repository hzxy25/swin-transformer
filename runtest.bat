set MASTER_ADDR=localhost
set MASTER_PORT=29500
set RANK=0
set WORLD_SIZE=1
set LOCAL_RANK=0

python main.py ^
  --cfg configs/swin/swin_base_patch4_window7_224.yaml ^
  --data-path "E:\tool\Data\WebFG-496-yolo" ^
  --batch-size 4 ^
  --output ./output_test ^
  --tag test_run