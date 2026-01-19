from deepchem.feat import MolecularFeaturizer
from rdkit.Chem import Descriptors

class ExternalDescriptorsRaw(MolecularFeaturizer):
    def __init__(self, exclude_descriptors=None, **kwargs):
        super().__init__(**kwargs)
        self.exclude = exclude_descriptors or []
        
        # Get all descriptor functions, excluding unwanted ones
        self.descriptor_names = []
        self.descriptor_funcs = []
        
        for desc_name, desc_func in Descriptors.descList:
            if desc_name not in self.exclude:
                self.descriptor_names.append(desc_name)
                self.descriptor_funcs.append(desc_func)
    
    def _featurize(self, mol):
        return [func(mol) for func in self.descriptor_funcs]
    
    def get_descriptor_names(self):
        return self.descriptor_names

# # Usage - exclude IPC and kappa3 descriptors
# exclude_list = ['Ipc', 'kappa3']
# featurizer = ExternalDescriptorsRaw(exclude_descriptors=exclude_list)