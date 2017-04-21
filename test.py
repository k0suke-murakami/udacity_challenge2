#:wq:
# Prediction code for the ai-world-team-c2
#
import os
import tensorflow as tf
import driving_data
import model
import scipy
import scipy.misc
import csv
import io
import gzip


OUTPUT_WIDTH = 160
OUTPUT_HIGH = 120
OUTPUT_CHAN = 3
IMAGES_DIR = '/root/sharefolder/sdc-data/ch2_001/center/'

#check
RESULT_FILE_S = '/root/sharefolder/sdc-data/ch2_001/result/n1_s.csv'
RESULT_FILE = '/root/sharefolder/sdc-data/ch2_001/result/n1.csv'
#check
CKPT_FILE = './best_1.01/model.ckpt'
#check
smoothing = True


imfiles = []
with open('/root/sharefolder/sdc-data/ch2_001/final_example.csv') as f:
    #reader = csv.reader(f)
    #buf = io.StringIO(f.read())
    #gzip_f = gzip.GzipFile(fileobj=f)
    #f = gzip_f.read()
    #f = unicode(f, 'utf-8')
    #print (f)
    #print (reader)
    #f = f.encode("utf-8")
    next(f, None)
    for line in sorted(f):
        #print (line[0:18])
        #imfiles.append(line.strip('\n')+'.jpg')
        imfiles.append(line[0:19]+'.jpg')
        #test.append(line[0:19]+'.jpg')
    #print (test)
#for (dirpath, dirnames, filenames) in os.walk(IMAGES_DIR):
#    for names in filenames:
#        imfiles.append(names)

#for root, dirs, files in os.walk("/home/aiwagan/Test"):
#    for file in files:
 #       if file.endswith('.png'):
 #           imfiles=imfiles+[file]




sess = tf.InteractiveSession()

saver = tf.train.Saver()


if os.path.isfile(CKPT_FILE):
    saver.restore(sess, CKPT_FILE)


steers = []

pre_pred = None


if smoothing == True:
    for f in imfiles:
        img_for_pred = scipy.misc.imresize(scipy.misc.imread(IMAGES_DIR+f)[240:, :], [120, 160])/255.0
        if pre_pred != None:
            img_for_forpred = pre_pred
        else:
            img_for_forpred = img_for_pred
        print(IMAGES_DIR+f)
    #print (img_for_pred.shape)
        pred_steer = model.y.eval(feed_dict={model.x: [img_for_pred], model.keep_prob: 1.0})[0][0]
        forpred_steer = model.y.eval(feed_dict={model.x: [img_for_forpred], model.keep_prob: 1.0})[0][0]
        print(pred_steer)
        steers.append((pred_steer+forpred_steer)/2.0)
        pre_pred = img_for_pred


    imgsteers = list(zip(imfiles, steers))
    with open(RESULT_FILE_S, 'w') as f:
        f.write("frame_id,steering_angle\n")
        for sublist in imgsteers:
            f.write(sublist[0][:-4]+',')
            f.write(str(sublist[1]))
            f.write("\n")



    smoothing = False

steers = []
if smoothing == False:
    for f in imfiles:
        img_for_pred = scipy.misc.imresize(scipy.misc.imread(IMAGES_DIR+f)[240:, :], [120, 160])/255.0
        print(IMAGES_DIR+f)
    #print (img_for_pred.shape)
        pred_steer = model.y.eval(feed_dict={model.x: [img_for_pred], model.keep_prob: 1.0})[0][0]
        print(pred_steer)
        steers.append(pred_steer)

#shuffle list of images
imgsteers = list(zip(imfiles, steers))



with open(RESULT_FILE, 'w') as f:
    f.write("frame_id,steering_angle\n")
    for sublist in imgsteers:
        f.write(sublist[0][:-4]+',')
        f.write(str(sublist[1]))
        f.write("\n")
