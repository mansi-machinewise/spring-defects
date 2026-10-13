from pathlib import Path

base = Path('D:/spring-defects/data/yolo_pose')

for split in ['train', 'val', 'test']:
    img_dir = base / split / 'images'
    lbl_dir = base / split / 'labels'
    
    imgs = set(p.stem for p in img_dir.iterdir() if p.suffix in ['.jpg','.jpeg','.png'])
    lbls = set(p.stem for p in lbl_dir.glob('*.txt'))
    
    matched  = imgs & lbls
    only_img = imgs - lbls
    only_lbl = lbls - imgs
    
    print(f'\n--- {split} ---')
    print(f'Images : {len(imgs)}')
    print(f'Labels : {len(lbls)}')
    print(f'Matched: {len(matched)}')
    if only_img:
        print(f'Images WITHOUT labels: {sorted(only_img)}')
    if only_lbl:
        print(f'Labels WITHOUT images: {sorted(only_lbl)}')
    if not only_img and not only_lbl:
        print('All matched perfectly.')
