from PIL import Image
from often_use import *

images = []
for i in [10, 20, 30]:
    # image_path = f'/home/yyoshitake/works/DeepSDF/project/sample_images/conpare_deptherr/only_deptherr/{i}.png'
    # image_path = f'/home/yyoshitake/works/DeepSDF/project/sample_images/conpare_deptherr/when_strong_deptherr/{i}.png'
    image_path = f'/home/yyoshitake/works/DeepSDF/project/sample_images/conpare_deptherr/wo_deptherr/{i}.png'
    img = Image.open(image_path)
    images.append(np.array(img)[360:740, 960:2100])
    
check_map_np(np.concatenate(images, axis=0), 'ccc.png')