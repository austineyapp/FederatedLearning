import torch
from torch import distributed as dist

from grace_dl.dist import Communicator


class Allgather(Communicator):
    def send_receive(self, tensors, name, ctx):
        self.size = 0
        if self.compressor.tensors_size_are_same:
            tensors_gathered = []
            for tensor_compressed in tensors:
                tensor_list = [torch.empty_like(tensor_compressed) for _ in range(self.world_size)]
                dist.all_gather(tensor_list, tensor_compressed)
                self.size += tensor_compressed.element_size() * tensor_compressed.nelement()
                tensors_gathered.append(tensor_list)
        else:
            local_sizes = torch.tensor([t.numel() for t in tensors]).cuda()  # TODO: set device
            gathered_sizes = [torch.empty_like(local_sizes) for _ in range(self.world_size)]
            dist.all_gather(gathered_sizes, local_sizes)  # tensor of tensor sizes per rank

            tensors_gathered = []
            for tensor, sizes in zip(tensors, zip(*gathered_sizes)):
                local_size = tensor.numel()
                max_size = max(sizes)
                gathered = []
                for _ in range(self.world_size):
                    padded = torch.empty(max_size, dtype=tensor.dtype, layout=tensor.layout, device=tensor.device)
                    gathered.append(padded)
                if local_size != max_size:
                    padding = torch.empty(max_size - local_size, dtype=tensor.dtype, layout=tensor.layout,
                                          device=tensor.device)
                    tensor = torch.cat((tensor, padding), dim=0)
                dist.all_gather(gathered, tensor)

                data_list = []
                for size, tensor_gathered in zip(sizes, gathered):
                    data_list.append(tensor_gathered[:size])

                tensors_gathered.append(data_list)

        # print("Compressed size: {:.1f}".format(self.size))
        decompressed_list = []
        for tensors_compressed in zip(*tensors_gathered):
            tensor_decompressed = self.compressor.decompress(tensors_compressed, ctx)
            decompressed_list.append(tensor_decompressed)
        tensors_aggregated = self.compressor.aggregate(decompressed_list)
        return (tensors_aggregated / self.world_size) if self.compressor.average else tensors_aggregated

    def acc(self, uncompressed_tensor):
        self.uncompressed_size += uncompressed_tensor.element_size() * uncompressed_tensor.nelement()

    def printr(self):
        print("Uncompressed")
        print(self.uncompressed_size)
        print("Compressed")
        print(self.size)
        print("Data Volume Compression Ratio: {:.1f}x".format(self.uncompressed_size/self.size))

