from aiogram import Bot, Dispatcher, types, F
from aiogram.filters.command import Command
import asyncio
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import Constants


bot = Bot(token=Constants.API_TOKEN)

dp = Dispatcher()
IMG_SIZE = 224
size = (IMG_SIZE, IMG_SIZE)

keras_model = load_model('model/Brain_model_best.h5')
keras_model.compile(optimizer=RMSprop(learning_rate=1e-4),
                    loss='sparse_categorical_crossentropy', metrics=['acc'])

datagen = ImageDataGenerator(rescale=1./255)


async def photo(file_id):
    file = await bot.get_file(file_id)
    file_path = file.file_path
    await bot.download_file(file_path, "user_image.jpg")
    df_test = pd.DataFrame({'image': ['user_image.jpg']})
    test_set = datagen.flow_from_dataframe(df_test,
                                           x_col='image',
                                           y_col=None,
                                           target_size=size,
                                           color_mode='grayscale',
                                           class_mode=None,
                                           batch_size=10,
                                           shuffle=False,
                                           interpolation='bilinear')

    predictions = keras_model.predict(test_set)
    predictions = predictions.argmax(axis=-1)
    pred = ['Affected' if x == 0 else 'Healthy' for x in predictions]
    return pred[0]


@dp.message(Command('start'))
async def send_welcome(message: types.Message):
    await message.answer('I will help you determine if there is a tumor on a brain MRI.')


@dp.message(F.content_type == 'document')
async def get_picture(message: types.Message):
    text = await (photo(message.document.file_id))
    await(message.answer(text))


@dp.message(F.content_type == 'photo')
async def get_picture(message: types.Message):
    text = await (photo(message.photo[-1].file_id))
    await(message.answer(text))


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
