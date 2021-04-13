class record_states:
    def __init__(self, outpath):
        self.outpath = outpath
        self.isopen = False
        
    def open(self):
        self.f = open(self.outpath, 'w')
        self.isopen = True
    
    def close(self):
        if self.isopen:
            self.f.close()
            
    def record(self, _, i, pkid, batch_x, model, *args, **kwargs):
        assert self.isopen
        self.f.write( pkid[0] + '\t' + ','.join(kwargs['k'].numpy().astype(str)) + '\n')
    
