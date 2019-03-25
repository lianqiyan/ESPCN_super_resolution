import cv2
import numpy as np
import tensorflow as tf
import argparse
from models.model_espcn import ESPCN
import os



def get_data(filename, color, resize, denoise):
    # file_name = 'test.jpg'
    im = cv2.imread(filename, 1)
    print(im.shape)


    r = 500.0 / im.shape[1]
    dim = (500, int(im.shape[0] * r))

    if not resize:
        # perform the actual resizing of the image and show it
        # resized = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)   # resize the image
        if denoise:
            im = cv2.medianBlur(im, 5)
        yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)  # rgb image to YCrCb format
    else:
        resized = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
        if denoise:
            resized = cv2.medianBlur(resized, 5)
        # resized = cv2.bilateralFilter(im, 9, 75, 75)
        yuv = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV)  # rgb image to YCrCb format

    # cv2.imwrite("thumbnail.png", resized)


    if not color:
        input = yuv[:, :, 0]  # get the Y channel file
        input = np.reshape(input, [1, input.shape[0], input.shape[1], 1]) / 255.0  # convert dtype to float64
        input = input.astype(np.float32)  # to float32

        cv2.imshow('image', yuv[:, :, 0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # print(input.dtype)
        # print(input.shape)

        dataset = tf.data.Dataset.from_tensor_slices(input)
        # print(dataset.output_types)
        # print(type(dataset))

        dataset = dataset.batch(1)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
    else:
        input = yuv  # get the Y channel file
        input = np.stack((input[:,:,0], input[:,:,1], input[:,:,2])) / 255.0  # convert dtype to float64
        input = input.astype(np.float32)  # to float32
        input = np.expand_dims(input, axis=3)
        print(input.shape)


        cv2.imshow('image', resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # print(input.dtype)
        # print(input.shape)

        dataset = tf.data.Dataset.from_tensor_slices(input)
        # print(dataset.output_types)
        # print(type(dataset))

        dataset = dataset.batch(3)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

    # with tf.Session() as sess:
    #     sess.run(iterator.initializer)
    #     temp = (sess.run(next_element))
    #     # temp = sess.run(dataset)
    #     print(type(temp))
    #     print(temp.shape)
    #     plt.imshow(temp[:, :, 0], 'gray')
    #     plt.show()
    return next_element, iterator


SAVE_NUM = 2
LOGDIR = 'evaluation_logdir/default'
STEPS_PER_LOG = 5

def get_arguments():
    parser = argparse.ArgumentParser(description='evaluate one of the models for image and video super-resolution')
    parser.add_argument('--model', type=str, default='espcn', choices=['srcnn', 'espcn', 'vespcn', 'vsrnet'],
                        help='What model to evaluate')
    parser.add_argument('--ckpt_path', default='checkpoints\espcn',
                        help='Path to the model checkpoint to evaluate')
    parser.add_argument('--scale_factor', default=2,
                        help='the model scale_factor')
    parser.add_argument('--save_num', type=int, default=SAVE_NUM,
                        help='How many images to write to summary')
    parser.add_argument('--steps_per_log', type=int, default=STEPS_PER_LOG,
                        help='How often to save image summaries')
    parser.add_argument('--use_mc', action='store_true',
                        help='Whether to use motion compensation in video super resolution model')
    parser.add_argument('--logdir', type=str, default=LOGDIR,
                        help='Where to save summaries')

    return parser.parse_args()



def super_resolution(model_args, filename, color, resize, denoise):
    model = ESPCN(model_args)

    with tf.Session() as sess:

        data_batch, data_iterator = get_data(filename, color, resize, denoise)

        print(data_batch.get_shape())
        predicted_batch = model.load_model(data_batch)

        if args.ckpt_path is None:
            print("Path to the checkpoint file was not provided")
            exit(1)

        if os.path.isdir(args.ckpt_path):
            args.ckpt_path = tf.train.latest_checkpoint(args.ckpt_path)

        print(args.ckpt_path)

        saver = tf.train.Saver()
        saver.restore(sess, args.ckpt_path)

        sess.run(data_iterator.initializer)
        # temp = sess.run(data_batch)
        # print(temp.shape)
        img = sess.run(predicted_batch)

        if color:
            img = np.dstack((img[0, :, :, :], img[1, :, :, :], img[2, :, :, :]))
            print(img.shape)
            rgb = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
            print(rgb.dtype)
            # print(rgb[0,::])

            cv2.imshow('image', rgb)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            rgb = rgb * 255.0  # back to uint8
            print(rgb.dtype)
            # print(rgb[0, ::])
            cv2.imwrite("sdog.png", rgb)
        else:
            print(type(img))
            print(img.shape)
            print(img.dtype)
            # img = img.astype(np.uint8)

            cv2.imshow('image', img[0, :, :, 0])
            cv2.waitKey(0)
            cv2.destroyAllWindows()



if __name__ == '__main__':
    # color = True
    ifcolor = True
    ifresize = True
    ifdenoise = True
    args = get_arguments()

    super_resolution(args, 'dog.jpg', ifcolor, ifresize, ifdenoise)






