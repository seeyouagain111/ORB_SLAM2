        
class VideoPipe(Pipeline):
        def __init__(self,batch_size,num_threads,device_id,data,shuffle):
            super(VideoPipe,self).__init__(batch_size, num_threads, device_id, seed=12)
            min_size=cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_size=cfg.DATA.TRAIN_JITTER_SCALES[1]
            size = int(round(np.random.uniform(min_size, max_size)))
            pos_x=round(np.random.uniform(0.2,0.8),1)
            pos_y=round(np.random.uniform(0.2,0.8),1)
            self.input=ops.VideoReader(device="gpu",file_root=data,sequence_length=cfg.DATA.NUM_FRAMES,shard_id=device_id,num_shards=cfg.NUM_GPUS,random_shuffle=shuffle,initial_fill=0,stride=cfg.DATA.SAMPLING_RATE)
            crop_size=cfg.DATA.TRAIN_CROP_SIZE
            self.resize=ops.Resize(device="gpu",resize_shorter=size)
            self.crop=ops.Crop(device="gpu",crop_h=crop_size,crop_w=crop_size,crop_pos_x=pos_x,crop_pos_y=pos_y)
            self.flip=ops.Flip(device="gpu")
            self.transpose=ops.Transpose(device="gpu",perm=[3,0,1,2])
            self.normalize=ops.Normalize(device="gpu",mean=0.45*255,stddev=0.225*255)
            self.A=np.random.random()
        def define_graph(self): 
            output,labels = self.input(name="Reader")
            fast_pathway=self.resize(output)
            fast_pathway=self.crop(fast_pathway)
            if self.A>0.5:
                fast_pathway=self.flip(fast_pathway)
            #fast_pathway=self.normalize(fast_pathway)   
            fast_pathway=self.transpose(fast_pathway)
            return fast_pathway,labels
