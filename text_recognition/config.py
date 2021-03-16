CHAR_VECTOR = '0123456789.- '

cfg = {
    'char_vector': CHAR_VECTOR,
    'class_num': len(CHAR_VECTOR)+1,

    'img_w': 128,
    'img_h': 64,
    'channel': 3,

    'train_batch_size': 32,
    'val_batch_size': 16,

    'downsample_factor': 4,
    'max_text_len': 5,
    
    'dataset_dir': 'data_generate/generated_imgs',

    'saved_model_path': 'custom_models/weights/VGGLSTM/VGGLSTM--002--0.582.hdf5'
}