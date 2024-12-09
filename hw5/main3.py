import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split

num_classes = 10
def load_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.17)
    return x_train, x_val, y_train, y_val, x_test, y_test
    
x_train, x_val, y_train, y_val, x_test, y_test = load_cifar10()
max_epochs = 20
batch_size = 64

def load_model(model_name):
    vgg, l = (tf.keras.applications.vgg19.VGG19, -10) if model_name == 'vgg19' else (tf.keras.applications.vgg16.VGG16, -8)
    model = vgg(weights='imagenet', include_top=False, classes=10, input_shape=(32,32,3))
    for layer in model.layers[:l]:
        layer.trainable = False
    x = tf.keras.layers.Flatten()(model.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=model.input, outputs=x)
    return model
    
def _fit(model_name):
    print(f'{model_name}:')
    model = load_model(model_name)
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=2, restore_best_weights=True, verbose=1)
    tbcb = keras.callbacks.TensorBoard(log_dir=f"logs/3/{model_name}", histogram_freq=0)
    model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9) ,metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=max_epochs, batch_size=batch_size, callbacks=[early_stopping, tbcb], verbose=1)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")


_fit("vgg16")
_fit("vgg19")
