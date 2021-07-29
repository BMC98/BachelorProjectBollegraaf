from keras.models import Model 
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras import regularizers 

img_height = 512
img_width = 512 
img_channels = 3 
classes = 5 

def build_unet_model(classes, img_height, img_width, img_channels):
	inputs = Input((img_width, img_height, img_channels))
	i = Lambda(lambda x: x / 255)(inputs) 

	#Contraction
	c1 = Conv2D(8, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same', kernel_regularizer = 'l2')(i)
	c1 = Dropout(0.1)(c1)
	c1 = Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer = 'l2')(c1)
	p1 = MaxPooling2D((2, 2))(c1)

	c2 = Conv2D(16, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same', kernel_regularizer = 'l2')(p1)
	c2 = Dropout(0.1)(c2)
	c2 = Conv2D(16, (3, 3),activation = 'relu', kernel_initializer = 'he_normal', padding = 'same', kernel_regularizer = 'l2')(c2)
	p2 = MaxPooling2D((2, 2))(c2)

	c3 = Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same', kernel_regularizer = 'l2')(p2)
	c3 = Dropout(0.1)(c3)
	c3 = Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same', kernel_regularizer = 'l2')(c3)
	p3 = MaxPooling2D((2,2))(c3)

	c4 = Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same', kernel_regularizer = 'l2')(p3)
	c4 = Dropout(0.1)(c4)
	c4 = Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same', kernel_regularizer = 'l2')(c4)
	p4 = MaxPooling2D((2,2))(c4)

	c5 = Conv2D(128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same', kernel_regularizer = 'l2')(p4)
	c5 = Dropout(0.1)(c5)
	c5 = Conv2D(128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same', kernel_regularizer = 'l2')(c5)

	#Expansion
	u6 = Conv2DTranspose(64, (2, 2), strides = (2, 2), padding = 'same', kernel_regularizer = 'l2')(c5)
	u6 = concatenate([u6, c4])
	c6 = Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same', kernel_regularizer = 'l2')(u6)
	c6 = Dropout(0.1)(c6)
	c6 = Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same', kernel_regularizer = 'l2')(c6)

	u7 = Conv2DTranspose(32, (2, 2), strides = (2, 2), padding = 'same', kernel_regularizer = 'l2')(c6)
	u7 = concatenate([u7, c3])
	c7 = Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same', kernel_regularizer = 'l2')(u7)
	c7 = Dropout(0.1)(c7)
	c7 = Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same', kernel_regularizer = 'l2')(c7)

	u8 = Conv2DTranspose(16, (2, 2), strides = (2, 2), padding = 'same', kernel_regularizer = 'l2')(c7)
	u8 = concatenate([u8, c2])
	c8 = Conv2D(16, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same', kernel_regularizer = 'l2')(u8)
	c8 = Dropout(0.1)(c8)
	c8 = Conv2D(16, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same', kernel_regularizer = 'l2')(c8)

	u9 = Conv2DTranspose(8, (2, 2), strides = (2, 2), padding = 'same', kernel_regularizer = 'l2')(c8)
	u9 = concatenate([u9, c1])
	c9 = Conv2D(8, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same', kernel_regularizer = 'l2')(u9)
	c9 = Dropout(0.1)(c9)
	c9 = Conv2D(8, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same', kernel_regularizer = 'l2')(c9)

	output = Conv2D(classes, (1, 1), activation = 'softmax')(c9)

	model = Model(inputs=[inputs], outputs=[output])

	#model.compile()

	return model 

