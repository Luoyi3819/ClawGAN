import cv2
from skimage.transform import resize
# from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, load_img
class DataLoader():
    def __init__(self, img_res=(256,256)):
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        path_sar = os.path.join('test_data/TM_CCORR_NORMED_infrared')  #320x240
        sar_list=os.listdir(path_sar)
        sar_list.sort()
        path_op=os.path.join('test_data/TM_CCORR_NORMED_visible')  #320x240
        op_list=os.listdir(path_op)
        op_list.sort()
        
        state = np.random.get_state()
        batch_images = np.random.choice(sar_list, size=batch_size)
        
        np.random.set_state(state)
        batch_labels = np.random.choice(op_list, size=batch_size)

        imgs_sar = []
        imgs_op = []
        
        for (sar_path,op_path) in zip(batch_images,batch_labels):
            print(sar_path,op_path)
            img_sar = img_to_array(load_img(os.path.join(path_sar,sar_path),color_mode='rgb'))
            img_op = img_to_array(load_img(os.path.join(path_op,op_path),color_mode='rgb'))

            #-----------------------模板匹配做裁剪-------------
            # img_sar = img_sar[20:220,30:300]
            # theight, twidth = img_sar.shape[:2]
            # result = cv2.matchTemplate(img_op,img_sar,cv2.TM_SQDIFF_NORMED)
            # #归一化处理
            # cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
            # #寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
            # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            # img_op = img_op[min_loc[1]:min_loc[1]+theight,min_loc[0]:min_loc[0]+twidth]
            #--------------------------------------------------
            img_sar = resize(img_sar, self.img_res)
            img_op = resize(img_op, self.img_res)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_sar = np.fliplr(img_sar)
                img_op = np.fliplr(img_op)
            
            imgs_sar.append(img_sar)
            imgs_op.append(img_op)

        imgs_sar = np.array(imgs_sar)/127.5 - 1.
        imgs_op = np.array(imgs_op)/127.5 - 1.
        #print('load_data_shape:',imgs_sar.shape)
        return imgs_op,imgs_sar

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        path_sar = os.path.join('train_data/TM_CCORR_NORMED_infrared')
        sar_list=os.listdir(path_sar)
        sar_list.sort()
        path_op=os.path.join('train_data/TM_CCORR_NORMED_visible')
        op_list=os.listdir(path_op)
        op_list.sort()
        self.n_batches = int(len(sar_list) / batch_size)

        for i in range(self.n_batches-1):
            batch_sar = sar_list[i*batch_size:(i+1)*batch_size]
            batch_op = op_list[i*batch_size:(i+1)*batch_size]
            imgs_sar, imgs_op = [], []
            for (img_sar,img_op) in zip(batch_sar,batch_op):
                img_sar = img_to_array(load_img(os.path.join(path_sar,img_sar),color_mode='rgb'))
                img_op=img_to_array(load_img(os.path.join(path_op,img_op),color_mode='rgb'))

                # img_sar = img_sar[20:220,30:300]
                # theight, twidth = img_sar.shape[:2]
                # result = cv2.matchTemplate(img_op,img_sar,cv2.TM_SQDIFF_NORMED)
                # #归一化处理
                # cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
                # #寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
                # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                # img_op = img_op[min_loc[1]:min_loc[1]+theight,min_loc[0]:min_loc[0]+twidth]

                img_sar = resize(img_sar, self.img_res)
                img_op = resize(img_op, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        img_sar = np.fliplr(img_sar)  #图像翻转
                        img_op = np.fliplr(img_op)

                imgs_sar.append(img_sar)
                imgs_op.append(img_op)

            imgs_sar = np.array(imgs_sar)/127.5 - 1.
            imgs_op = np.array(imgs_op)/127.5 - 1.
            # print(imgs_sar.shape)
            yield imgs_op,imgs_sar
    
    def load_batch_tricks(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        path_sar = os.path.join('train_data/infrared')
        sar_list=os.listdir(path_sar)
        sar_list.sort()
        path_op=os.path.join('train_data/visible')
        op_list=os.listdir(path_op)
        op_list.sort()
        self.n_batches = int(len(sar_list) / batch_size)

        for i in range(self.n_batches-1):
            batch_sar = sar_list[i*batch_size:(i+1)*batch_size]
            batch_op = op_list[i*batch_size:(i+1)*batch_size]
            imgs_sar, imgs_op = [], []
            for (img_sar,img_op) in zip(batch_sar,batch_op):
                img_sar = img_to_array(load_img(os.path.join(path_sar,img_sar),color_mode='rgb'))
                img_op=img_to_array(load_img(os.path.join(path_op,img_op),color_mode='rgb'))

                # img_sar = img_sar[20:220,30:300]
                # theight, twidth = img_sar.shape[:2]
                # result = cv2.matchTemplate(img_op,img_sar,cv2.TM_SQDIFF_NORMED)
                # #归一化处理
                # cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
                # #寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
                # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                # img_op = img_op[min_loc[1]:min_loc[1]+theight,min_loc[0]:min_loc[0]+twidth]

                img_sar = resize(img_sar, self.img_res)
                img_op = resize(img_op, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        img_sar = np.fliplr(img_sar)  #图像翻转
                        img_op = np.fliplr(img_op)

                imgs_sar.append(img_sar)
                imgs_op.append(img_op)

            imgs_sar = np.array(imgs_sar)/127.5 - 1.
            imgs_op = np.array(imgs_op)/127.5 - 1.
            # print(imgs_sar.shape)
            yield imgs_op,imgs_sar
    # def imread(self, path):
    #     return scipy.misc.imread(path, mode='RGB').astype(np.float)

if __name__ == '__main__':
    os.makedirs('test_dataloder/', exist_ok=True)
    r, c = 3,2
    datalo=DataLoader()
    imgs_A, imgs_B = datalo.load_data(batch_size=3, is_testing=True)

    # gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

    # Rescale images 0 - 1
    # gen_imgs = 0.5 * gen_imgs + 0.5
    imgs_B =0.5 * imgs_B + 0.5
    imgs_A=0.5 * imgs_A + 0.5
    print('sample',imgs_B.shape,imgs_A.shape)
    gen_imgs = [imgs_B,imgs_A]

    titles = ['Condition', 'Original']
    fig, axs = plt.subplots(r, c)
    for i in range(r): #batch
        for j in range(c):
            if j ==0:
                axs[i,j].imshow(gen_imgs[j][i][:,:,0],cmap='gray')
            else:
                axs[i,j].imshow(gen_imgs[j][i])
            axs[i, j].set_title(titles[j])
            axs[i,j].axis('off')
    fig.savefig("test_dataloder/testload.png")
    plt.close()
        
