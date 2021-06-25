from skimage import color, io, exposure, filters
from skimage import img_as_float
from PIL import Image, ImageFont, ImageDraw
from skimage.transform import rescale
import numpy as np
import json

def get_cropped_img(img, target_res):
    h, w = img.shape[:2]
    h_target, w_target = target_res

    h_start = (h-h_target)//2
    h_end = h_start + h_target

    w_start = (w-w_target)//2
    w_end = w_start + w_target

    target_img = img[h_start:h_end, w_start:w_end]
    return target_img

def add_text(img, text, font, align='center', **kwargs):
    font_size = kwargs.get('font_size', 16)
    font_color = kwargs.get('font_color', [255, 255, 255])
    origin = kwargs.get('origin', (0, 0))

    font_color = tuple(font_color)

    pil_img = Image.fromarray(np.uint8(img*255))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font, font_size)
    draw.text(origin, text, font_color, font=font, align=align)
    out_img = np.array(pil_img) / 255
    return out_img

def txt_formatter(text, **kwargs):
    maxlen = kwargs.get('maxlen', 0)
    assert type(maxlen)==int, 'maxlen must be integer'
    assert maxlen>0, 'maxlen must be >= 0'

    words = text.split(' ')
    lines = []
    builder = ''
    for word in words:
        if builder == '':
            builder = word
        elif (len(builder) + len(word)) + 1 <= kwargs['maxlen']:
            builder = ' '.join([builder, word])
        else:
            lines.append(builder)
            builder = word
    lines.append(builder)

    return '\n'.join(lines), len(lines)

def get_origin(imsize, maxlen, font_size, num_lines, w_unit=36):
    h_unit = font_size
    w_space = w_unit * maxlen
    h_space = h_unit * num_lines
    h_origin = (imsize[0] - h_space) // 2
    w_origin = (imsize[1] - w_space) // 2
    return (w_origin, h_origin)

def apply_mask(img, mask_color, alpha):
    color_mask = np.zeros_like(img)
    mask_color = np.array(mask_color)
    mask_color = mask_color / 255
    color_mask[:, :] = mask_color

    img_hsv = color.rgb2hsv(img)
    color_mask_hsv = color.rgb2hsv(color_mask)

    print('Converted to HSV')

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    print('Mask applied')

    img_masked = color.hsv2rgb(img_hsv)

    print('Converted back to RGB')

    return img_masked

def apply_mask_alt(img, mask_color):
    mask_color = np.array(mask_color) / 255
    img_mask = np.minimum(img, mask_color)

    print('Alt mask applied')

    return img_mask


def save_colored_images(config_loc):
    with open(config_loc, 'r') as f:
        config = json.load(f)

    print('Config loaded')

    im = io.imread(config['img_path']+config['img_name']+'.'+config['img_ext'])
    img = img_as_float(im)
    img_rgb = img[:, :, :3]

    print(f'Image loaded. Shape: {img_rgb.shape}')

    if config['rescale']>0:
        img_rgb = rescale(img_rgb, config['rescale'], multichannel=True)
        print(f'Image rescaled. New shape: {img_rgb.shape}')

    if config['alt_mask']:
        img_masked = apply_mask_alt(img_rgb, config['mask_color'])
    else:
        img_masked = apply_mask(img_rgb, config['mask_color'], config['alpha'])

    if config['gamma'] >= 0:
        img_masked = exposure.adjust_gamma(img_masked, config['gamma'])
        print('Applied gamma correction')
    if config['blur_strength'] > 0:
        img_masked = filters.gaussian(img_masked, sigma=config['blur_strength'], multichannel=True)

    img_card = get_cropped_img(img_masked, config['card_res'])
    img_header = get_cropped_img(img_masked, config['header_res'])

    img_card_name = f'{config["output_name"]}_{config["card_suffix"]}.{config["img_ext"]}'
    img_header_name = f'{config["output_name"]}_{config["header_suffix"]}.{config["img_ext"]}'

    print(f'Card image created. Name: {img_card_name}. Shape: {img_card.shape}')
    print(f'Header image created. Name: {img_header_name}. Shape: {img_header.shape}')

    if config['add_text']:
        text_cfg = config['text_cfg']
        if text_cfg['auto_format']:
            txt, num_lines = txt_formatter(text_cfg['text'], maxlen=text_cfg['maxlen'])
        else:
            txt, num_lines = text_cfg['text'], text_cfg['num_lines']
        origin = get_origin(img_card.shape[:2], text_cfg['maxlen'], text_cfg['font_size'], num_lines, w_unit=text_cfg['w_unit'])
        img_card = add_text(img_card, txt, font=text_cfg['font'], font_size=text_cfg['font_size'], font_color=text_cfg['font_color'], origin=origin)
        print(f'Text added to card. Num lines: {num_lines}')

    img_card = np.uint8(img_card*255)
    img_header = np.uint8(img_header*255)

    io.imsave(config['save_path']+img_card_name, img_card)
    io.imsave(config['save_path']+img_header_name, img_header)

    print('Images saved')



if __name__ == '__main__':
    save_colored_images('_patterns/args.json')
    print('Done.')