import tensorflow as tf

def get_encoder(input_shape, is_trainable=False):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = is_trainable
    # down_stack.trainable = True
    return down_stack
    
def upsample(filters, size):
    """Upsamples an input.
        Conv2DTranspose => Batchnorm => Relu
        Args:
        filters: number of filters
        size: filter size
        Returns:
        Upsample Functional Model
    """
    def apply(input):
        initializer = tf.random_normal_initializer(0., 0.02)

        x = tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                          padding='same',
                                          kernel_initializer=initializer,
                                          use_bias=False)(input)

        x = tf.keras.layers.BatchNormalization()(x)

        
        x = tf.keras.layers.ReLU()(x)
        return x

    return apply
    
def unet_model(output_channels):
    encoder = get_encoder(input_shape=[256, 256, 3], is_trainable = False)

    up_stack = [
        upsample(512, 3),  
        upsample(256, 3),  
        upsample(128, 3),  
        upsample(64, 3),   
    ]
    
    # Downsampling through the model
    skips = encoder.outputs
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same') 

    x = last(x)
    x = tf.keras.layers.Activation('sigmoid')(x)

    return tf.keras.Model(inputs=encoder.inputs, outputs=x)