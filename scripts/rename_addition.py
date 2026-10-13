import os
from pathlib import Path

# rename train_addition: spring_201.jpg onwards
train_dir = Path('C:/Users/mansi/Downloads/train_addition')
for i, img in enumerate(sorted(train_dir.glob('*.jpg')), start=201):
    new = train_dir / f'spring_{i:03d}.jpg'
    os.rename(img, new)
    print(f'{img.name} -> {new.name}')
print(f'Train done: {i-200} images renamed.')

# rename test_addition: spring_015.jpg onwards
test_dir = Path('C:/Users/mansi/Downloads/test_addition')
for i, img in enumerate(sorted(test_dir.glob('*.jpg')), start=15):
    new = test_dir / f'spring_{i:03d}.jpg'
    os.rename(img, new)
    print(f'{img.name} -> {new.name}')
print(f'Test done: {i-14} images renamed.')
