
from .resource_manager import ResourceManager
import mindspore.dataset as ds
import os
from mindspore.communication.management import get_rank, get_group_size
class GenerDataSet():
    def __init__(self, resource_manager: ResourceManager, for_train=True, dtype='fr_test'):
        self.for_train = for_train
        self.dtype = dtype

        #self.to_tensor = transforms.ToTensor()
        self.resource_manager = resource_manager
        batch_size = 16
        if dtype == 'train':
            self.count = len(self.resource_manager.train_blocks)
        elif dtype == 'test':
            self.count = len(self.resource_manager.test_blocks)
            volunteers_nums = batch_size-self.count%batch_size
            print(volunteers_nums,self.count)
            if volunteers_nums != batch_size:
                self.resource_manager.test_blocks += self.resource_manager.test_blocks[:volunteers_nums]
                print(len(self.resource_manager.test_blocks))
            self.count = len(self.resource_manager.test_blocks)
        else:
            self.count = len(self.resource_manager.fr_test_blocks)
            volunteers_nums = batch_size-self.count%batch_size
            print(volunteers_nums,self.count)
            if volunteers_nums != batch_size:
                self.resource_manager.fr_test_blocks += self.resource_manager.fr_test_blocks[:volunteers_nums]
                print(len(self.resource_manager.fr_test_blocks))
            self.count = len(self.resource_manager.fr_test_blocks)
    def __getitem__(self, index):
        """return input, label"""
        
        if self.dtype == 'train':
            input, label = self.resource_manager.train_blocks[index]
        elif self.dtype == 'test':
 
            input, label = self.resource_manager.test_blocks[index]
            
        else:
            input, label = self.resource_manager.fr_test_blocks[index]

        #return [self.to_tensor(item) for item in input], self.to_tensor(label)
        return input[0].transpose((2,0,1)),input[1].transpose((2,0,1)),input[2].transpose((2,0,1)),input[3].transpose((2,0,1)),label.transpose((2,0,1))

    def __len__(self):
        return self.count


def create_train_dataset(dataset,batch=16):
    '''
      Create train dataset.
    '''

    device_num, rank_id = _get_rank_info()
    print(device_num)
    if device_num == 1 :

        train_ds = ds.GeneratorDataset(dataset, column_names=["input_images0","input_images1","input_images2","input_images3", "target_images"], shuffle=False,num_parallel_workers=8)
    else:
        shard_id = get_rank() 
        num_shards = get_group_size()
        train_ds = ds.GeneratorDataset(dataset, column_names=["input_images0","input_images1","input_images2","input_images3", "target_images"], shuffle=False,num_parallel_workers=1,
                                            num_shards=num_shards, shard_id=shard_id)

    train_ds = train_ds.batch(batch, drop_remainder=True)

    return train_ds
    
def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        from mindspore.communication.management import get_rank, get_group_size
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = 1
        rank_id = None
    return rank_size, rank_id

class get_dataset():
    def __init__(self,data_root=r'/home/ma-user/work/{}',train='train',batch=16):
        self.train_dataset = create_train_dataset(GenerDataSet(ResourceManager(resource=data_root),dtype=train),batch)





