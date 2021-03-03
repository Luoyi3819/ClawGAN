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
        self.test_infrared_dir = 'testdata/infrared'
        self.test_visible_dir = 'testdata/visible'
        self.train_infrared_dir = 'traindata/infrared'
        self.train_visible_dir = 'traindata/visible'

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        path_inf = os.path.join(self.test_infrared_dir)  #320x240
        inf_list=os.listdir(path_inf)
        inf_list.sort()
        path_vis=os.path.join(self.test_visible_dir)  #320x240
        vis_list=os.listdir(path_vis)
        vis_list.sort()
        
        state = np.random.get_state()
        batch_images = np.random.choice(inf_list, size=batch_size)
        
        np.random.set_state(state)
        batch_labels = np.random.choice(vis_list, size=batch_size)

        imgs_inf = []
        imgs_vis = []
        
        for (inf_path,vis_path) in zip(batch_images,batch_labels):
            print(inf_path,vis_path)
            img_inf = img_to_array(load_img(os.path.join(path_inf,inf_path),color_mode='rgb'))
            img_vis = img_to_array(load_img(os.path.join(path_vis,vis_path),color_mode='rgb'))

            #-----------------------MACTH TEMPLATE-------------
            img_inf = img_inf[20:220,30:300]
            theight, twidth = img_inf.shape[:2]
            result = cv2.matchTemplate(img_vis,img_inf,cv2.TM_SQDIFF_NORMED)
            #normalize
            cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )          
            #find the position of max/min values from max 
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            img_vis = img_vis[min_loc[1]:min_loc[1]+theight,min_loc[0]:min_loc[0]+twidth]
            #--------------------------------------------------
            
            img_inf = resize(img_inf, self.img_res)
            img_vis = resize(img_vis, self.img_res)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_inf = np.fliplr(img_inf)
                img_vis = np.fliplr(img_vis)
            
            imgs_inf.append(img_inf)
            imgs_vis.append(img_vis)

        imgs_inf = np.array(imgs_inf)/127.5 - 1.
        imgs_vis = np.array(imgs_vis)/127.5 - 1.
        #print('load_data_shape:',imgs_inf.shape)
        return imgs_vis,imgs_inf

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        path_inf = os.path.join(self.train_infrared_dir)
        inf_list=os.listdir(path_inf)
        inf_list.sort()
        path_vis=os.path.join(self.train_visible_dir)
        vis_list=os.listdir(path_vis)
        vis_list.sort()
        self.n_batches = int(len(inf_list) / batch_size)

        for i in range(self.n_batches-1):
            batch_inf = inf_list[i*batch_size:(i+1)*batch_size]
            batch_vis = vis_list[i*batch_size:(i+1)*batch_size]
            imgs_inf, imgs_vis = [], []
            for (img_inf,img_vis) in zip(batch_inf,batch_vis):
                img_inf = img_to_array(load_img(os.path.join(path_inf,img_inf),color_mode='rgb'))
                img_vis=img_to_array(load_img(os.path.join(path_vis,img_vis),color_mode='rgb'))

                #-----------------------MACTH TEMPLATE-------------
	            img_inf = img_inf[20:220,30:300]
	            theight, twidth = img_inf.shape[:2]
	            result = cv2.matchTemplate(img_vis,img_inf,cv2.TM_SQDIFF_NORMED)
	            #normalize
	            cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )          
	            #find the position of max/min values from max 
	            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
	            img_vis = img_vis[min_loc[1]:min_loc[1]+theight,min_loc[0]:min_loc[0]+twidth]
	            #--------------------------------------------------

                img_inf = resize(img_inf, self.img_res)
                img_vis = resize(img_vis, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        img_inf = np.fliplr(img_inf)  #image flip
                        img_vis = np.fliplr(img_vis)

                imgs_inf.append(img_inf)
                imgs_vis.append(img_vis)

            imgs_inf = np.array(imgs_inf)/127.5 - 1.
            imgs_vis = np.array(imgs_vis)/127.5 - 1.
            # print(imgs_inf.shape)
            yield imgs_vis,imgs_inf

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
        
