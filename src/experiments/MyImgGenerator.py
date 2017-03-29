from keras.preprocessing.image import DirectoryIterator



class MyImgGenerator(object):

    

    def __init__(self, gen):
        self.gen = gen


    def next(self):
        imgs = self.gen.next()
        
        return (imgs,imgs)
