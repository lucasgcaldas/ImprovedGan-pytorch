import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def plot_img(img, title):
    """
    Plot an image.

    Parameters:
        img (numpy.ndarray): The input image.
        title (str): The title of the plot.

    Returns:
        None
    """
    plt.figure()
    plt.imshow(img, interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

def img_stretch(img):
    """
    Stretch the pixel values of an image to the range [0, 1].

    Parameters:
        img (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The image with pixel values stretched to [0, 1].
    """
    img = img.astype(float)
    img -= np.min(img)
    img /= np.max(img) + 1e-12
    return img

def img_tile(imgs, aspect_ratio=1.0, tile_shape=None, border=1,
             border_color=0, stretch=False):
    '''
    Tile images in a grid.
    If tile_shape is provided only as many images as specified in tile_shape
    will be included in the output.

    Parameters:
        imgs (numpy.ndarray): The images to be tiled.
        aspect_ratio (float): The aspect ratio of each tile.
        tile_shape (tuple): The desired shape of the tile grid.
        border (int): The width of the border between tiles.
        border_color (int): The color of the border.
        stretch (bool): Whether to stretch the pixel values of the images.

    Returns:
        numpy.ndarray: The tiled image grid.
    '''

    # Prepare images
    if stretch:
        imgs = img_stretch(imgs)
    imgs = np.array(imgs)
    if imgs.ndim != 3 and imgs.ndim != 4:
        raise ValueError('imgs has wrong number of dimensions.')
    n_imgs = imgs.shape[0]

    # Grid shape
    img_shape = np.array(imgs.shape[1:3])
    if tile_shape is None:
        img_aspect_ratio = img_shape[1] / float(img_shape[0])
        aspect_ratio *= img_aspect_ratio
        tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
        tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
        grid_shape = np.array((tile_height, tile_width))
    else:
        assert len(tile_shape) == 2
        grid_shape = np.array(tile_shape)

    # Tile image shape
    tile_img_shape = np.array(imgs.shape[1:])
    tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

    # Assemble tile image
    tile_img = np.empty(tile_img_shape)
    tile_img[:] = border_color
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            img_idx = j + i * grid_shape[1]
            if img_idx >= n_imgs:
                # No more images - stop filling out the grid.
                break
            img = imgs[img_idx]
            yoff = (img_shape[0] + border) * i
            xoff = (img_shape[1] + border) * j
            tile_img[yoff:yoff + img_shape[0], xoff:xoff + img_shape[1], ...] = img

    return tile_img

def conv_filter_tile(filters):
    """
    Tile convolutional filters into a grid.

    Parameters:
        filters (numpy.ndarray): The filters to be tiled.

    Returns:
        numpy.ndarray: The tiled filter grid.
    """
    n_filters, n_channels, height, width = filters.shape
    tile_shape = None
    if n_channels == 3:
        filters = np.transpose(filters, (0, 2, 3, 1))
    else:
        tile_shape = (n_channels, n_filters)
        filters = np.transpose(filters, (1, 0, 2, 3))
        filters = np.resize(filters, (n_filters * n_channels, height, width))
    filters = img_stretch(filters)
    return img_tile(filters, tile_shape=tile_shape)

def scale_to_unit_interval(ndar, eps=1e-8):
    """
    Scale all values in the ndarray to be between 0 and 1.

    Parameters:
        ndar (numpy.ndarray): The input array.
        eps (float): A small value added to the denominator for numerical stability.

    Returns:
        numpy.ndarray: The scaled array between 0 and 1.
    """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row into an array in which images are reshaped and laid out like tiles on a floor.

    Parameters:
        X (numpy.ndarray or tuple): The input array or a tuple of 4 channels.
        img_shape (tuple): The original shape of each image.
        tile_shape (tuple): The number of images to tile (rows, cols).
        tile_spacing (tuple): The spacing between tiles (vertically, horizontally).
        scale_rows_to_unit_interval (bool): Whether to scale the values to [0, 1].
        output_pixel_vals (bool): Whether the output should be pixel values (int8) or floats.

    Returns:
        numpy.ndarray: An array suitable for viewing as an image.
    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                 in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                out_array[:, :, i] = np.zeros(out_shape,
                                              dtype='uint8' if output_pixel_vals else out_array.dtype) + channel_defaults[i]
            else:
                out_array[:, :, i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing,
                                                        scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        H, W = img_shape
        Hs, Ws = tile_spacing
        out_array = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale_rows_to_unit_interval:
                        this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                    else:
                        this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * (255 if output_pixel_vals else 1)
        return out_array
