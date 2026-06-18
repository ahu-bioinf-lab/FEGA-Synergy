from torch_geometric.data import Data



class CustomData(Data):

        # Since we have converted the node graph to the line graph, we should specify the increase of the index as well.
        '''
        def __inc__(self, key, value, *args, **kwargs):
        # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
        # Replace with "def __inc__(self, key, value, *args, **kwargs)"
            if key == 'line_graph_edge_index':
                return self.edge_index.size(1) if self.edge_index.nelement()!=0 else 0
            return super().__inc__(key, value, *args, **kwargs)'''

        # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
        # Replace with "return super().__inc__(self, key, value, args, kwargs)"
        def __inc__(self, key, value, *args, **kwargs):
            if key == 'line_graph_edge_index':
                return value.size(1) if value.dim() == 2 else 0
            return super().__inc__(key, value, *args, **kwargs)