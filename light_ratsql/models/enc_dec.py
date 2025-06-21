import torch
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np

from ratsql.models import abstract_preproc
from ratsql.utils import registry

class ZippedDataset(torch.utils.data.Dataset):
    def __init__(self, *components):
        assert len(components) >= 1
        lengths = [len(c) for c in components]
        assert all(lengths[0] == other for other in lengths[1:]), f"Lengths don't match: {lengths}"
        self.components = components
    
    def __getitem__(self, idx):
        return tuple(c[idx] for c in self.components)
    
    def __len__(self):
        return len(self.components[0])


@registry.register('model', 'EncDec')
class EncDecModel(torch.nn.Module):
    class Preproc(abstract_preproc.AbstractPreproc):
        def __init__(
                self,
                encoder,
                decoder,
                encoder_preproc,
                decoder_preproc):
            super().__init__()

            self.enc_preproc = registry.lookup('encoder', encoder['name']).Preproc(**encoder_preproc)
            self.dec_preproc = registry.lookup('decoder', decoder['name']).Preproc(**decoder_preproc)
        
        def validate_item(self, item, section):
            enc_result, enc_info = self.enc_preproc.validate_item(item, section)
            dec_result, dec_info = self.dec_preproc.validate_item(item, section)
            
            return enc_result and dec_result, (enc_info, dec_info)
        
        def add_item(self, item, section, validation_info):
            enc_info, dec_info = validation_info
            self.enc_preproc.add_item(item, section, enc_info)
            self.dec_preproc.add_item(item, section, dec_info)
        
        def clear_items(self):
            self.enc_preproc.clear_items()
            self.dec_preproc.clear_items()

        def save(self):
            self.enc_preproc.save()
            self.dec_preproc.save()
        
        def load(self):
            self.enc_preproc.load()
            self.dec_preproc.load()
        
        def dataset(self, section):
            return ZippedDataset(self.enc_preproc.dataset(section), self.dec_preproc.dataset(section))
        
    def __init__(self, preproc, device, encoder, decoder):
        super().__init__()
        self.i=0
        self.preproc = preproc
        self.encoder = registry.construct(
                'encoder', encoder, device=device, preproc=preproc.enc_preproc)
        self.decoder = registry.construct(
                'decoder', decoder, device=device, preproc=preproc.dec_preproc)
        
        if getattr(self.encoder, 'batched'):
            self.compute_loss = self._compute_loss_enc_batched
        else:
            self.compute_loss = self._compute_loss_unbatched

    def _compute_loss_enc_batched(self, batch, debug=False):
        losses = []
        
        enc_states = self.encoder([enc_input for enc_input, dec_output in batch])
        
        for enc_state, (enc_input, dec_output) in zip(enc_states, batch):
            loss = self.decoder.compute_loss(enc_input, dec_output, enc_state, debug)
            losses.append(loss)
            #print('ortho_loss', enc_state)
        if debug:
            return losses
        else:
            return torch.mean(torch.stack(losses, dim=0), dim=0)

    def _compute_loss_enc_batched2(self, batch, debug=False):
        losses = []
        print('test10')
        for enc_input, dec_output in batch:
            enc_state, = self.encoder([enc_input])
            loss = self.decoder.compute_loss(enc_input, dec_output, enc_state, debug)
            losses.append(loss)
        if debug:
            return losses
        else:
            return torch.mean(torch.stack(losses, dim=0), dim=0)

    def _compute_loss_unbatched(self, batch, debug=False):
        losses = []
        print('test20')
        for enc_input, dec_output in batch:
            enc_state = self.encoder(enc_input)
            loss = self.decoder.compute_loss(enc_input, dec_output, enc_state, debug)
            losses.append(loss)
        if debug:
            return losses
        else:
            return torch.mean(torch.stack(losses, dim=0), dim=0)

    def eval_on_batch(self, batch):
        mean_loss = self.compute_loss(batch).item()
        batch_size = len(batch)
        result = {'loss': mean_loss * batch_size, 'total': batch_size}
        return result

    def begin_inference(self, orig_item, preproc_item):
        enc_input, _ = preproc_item
        if getattr(self.encoder, 'batched'):
            enc_state, = self.encoder([enc_input])
        else:
            enc_state = self.encoder(enc_input)
        self.plot_grat_ok(enc_input, enc_state)

        return self.decoder.begin_inference(enc_state, orig_item)
    
    def listToString(self, s,type):
        str1 = type
        i=0
        for ele in s:
            if(type=='col::' and i==0):
                i=i+1
            else:
                str1 += ele+'_'
        return str1[0:len(str1)-1]

    def plot_grat_ok(self, enc_input, enc_state):
        self.i=self.i+1
        tab=np.array(enc_state.m2t_align_mat)
        col=np.array(enc_state.m2c_align_mat)

        #temp=torch.cat(( col, tab), 1)
        #a,b=temp.shape
        #temp=temp.reshape(a,b)

        question=enc_input['question']
        tab2=enc_input['tables']
        col2=enc_input['columns']

        tables =list()
        for i in tab2 :
            tables.append(self.listToString(i, "table::"))

        columns =list()
        for i in col2 :
            columns.append(self.listToString(i, "col::"))

        schema=columns+tables
        all=question+schema

        plt.rcParams["figure.figsize"] = [15,15]
        plt.rcParams["figure.autolayout"] = True


        fig, ax = plt.subplots(1,2)

        #fig.subplots_adjust(left=0.05, right=0.95, wspace=0.3)

        # Set size of first subplot to [5, 5]
        #fig.axes[0].set_position([0.05, 0.05, 0.4, 0.9])
        
        # Set size of second subplot to [15, 5]
        #fig.axes[1].set_position([0.55, 0.05, 0.4, 0.9])
        
        #plt.imshow(np.array(temp), cmap='gist_gray', interpolation='nearest')
        matrix=np.array(tab[:,0:len(tables)])
        #matrix2=np.array(col[0:len(question),0:len(columns)])
        matrix2=np.array(col[:,0:len(columns)])

        ax[0].matshow(matrix, cmap='gist_gray', interpolation='nearest', aspect='auto')
        ax[1].matshow(matrix2, cmap='gist_gray', interpolation='nearest', aspect='auto')
        #plt.xticks(ticks=schema)

        fig.subplots_adjust(left=0.05, right=0.95, wspace=0.3)

        #ax.set(xlim =(len(schema),0), ylim =(len(question),0))

        for i in range(len(all)):
            ax[0].text(len (tables),i, str(all[i]), va='center', ha='left')

        for i in range(len(tables)):
            ax[0].text(i,len(all), str(tables[i]), va='top', ha='center', rotation=75)
        
        for i in range(len(columns)):
            ax[1].text(i,len(all), str(columns[i]), va='top', ha='center', rotation=75)

        #plt.show()

        #ax[0].set_axis_off()
        #ax[1].set_axis_off()
        #ax[1].set_position([0.5, 0.1, 0.1, 0.1])

        plt.savefig("att_"+f"{self.i}"+".png")


    def plot_grat2(self, enc_input, enc_state):
        self.i=self.i+1
        tab=np.array(enc_state.m2t_align_mat)
        col=np.array(enc_state.m2c_align_mat)

        question=enc_input['question']
        tab2=enc_input['tables']
        col2=enc_input['columns']

        tables =list()
        for i in tab2 :
            tables.append(self.listToString(i, "table::"))

        columns =list()
        for i in col2 :
            columns.append(self.listToString(i, "col::"))

        schema=columns+tables
        all=question+schema

        plt.rcParams["figure.figsize"] = [15,15]
        plt.rcParams["figure.autolayout"] = True


        #fig, ax = plt.subplots(1,2)

        #fig.subplots_adjust(left=0.05, right=0.95, wspace=0.3)

        # Set size of first subplot to [5, 5]
        #fig.axes[0].set_position([0.05, 0.05, 0.4, 0.9])

        # Set size of second subplot to [15, 5]
        #fig.axes[1].set_position([0.55, 0.05, 0.4, 0.9])

        #plt.imshow(np.array(temp), cmap='gist_gray', interpolation='nearest')
        matrix=np.concatenate((tab, col), axis=-1)
        

        #ax[0].matshow(matrix, cmap='gist_gray', interpolation='nearest', aspect='auto')
        plt.imshow(matrix[:, 0:len(question)], cmap='gist_gray', interpolation='nearest')


        #ax[1].matshow(matrix2, cmap='gist_gray', interpolation='nearest', aspect='auto')
        x_labels =[schema]
        
        plt.xticks(range(len(x_labels)), x_labels)
        y_labels = [question]
        plt.yticks(range(len(y_labels)), y_labels)

        #fig.subplots_adjust(left=0.05, right=0.95, wspace=0.3)

        #ax.set(xlim =(len(schema),0), ylim =(len(question),0))

        #for i in range(len(question)):
        #    ax[0].text(len (tables),i, str(question[i]), va='center', ha='left')

        #for i in range(len(schema)):
        #    ax[0].text(i,len(schema), str(schema[i]), va='top', ha='center', rotation=75)

        #for i in range(len(columns)):
        #    ax[1].text(i,len(all), str(columns[i]), va='top', ha='center', rotation=75)

        #plt.show()

        #ax[0].set_axis_off()
        #ax[1].set_axis_off()
        #ax[1].set_position([0.5, 0.1, 0.1, 0.1])

        plt.savefig("att_"+f"{self.i}"+".png")




    def plot_grat(self, enc_input, enc_state):
        self.i = self.i + 1
        tab = np.array(enc_state.m2t_align_mat)
        col = np.array(enc_state.m2c_align_mat)

        temp = np.concatenate((col, tab), 1)
        a, b = temp.shape
        temp = temp.reshape(a, b)

        question = enc_input['question']
        tab2 = enc_input['tables']
        col2 = enc_input['columns']

        tables = list()
        for i in tab2:
            tables.append(self.listToString(i, "table::"))

        columns = list()
        for i in col2:
            columns.append(self.listToString(i, "col::"))

        schema = columns + tables
        all_labels = question + schema

        # Plot the attention matrix (temp) with labels
        plt.figure(figsize=(15, 10))
        plt.imshow(temp[0:len(question)], cmap='viridis', aspect='auto')
        plt.colorbar()

        # Set ticks and labels for both x and y axes
        plt.xticks(range(len(schema)), schema, rotation=75, fontsize=10,ha='center',va='top')
        plt.yticks(range(len(question)), question, fontsize=10)
        
        # Set axis labels
        plt.xlabel('Schema', fontsize=12)
        plt.ylabel('Questions', fontsize=12)

        plt.title('Alignment between the question and the database schema', fontsize=15)

        plt.tight_layout()

        # Save the plot to a file
        plt.savefig("att_" + f"{self.i}" + ".png")

        if (self.i==16677776):
            #data=torch.from_numpy(temp)
            data, value=enc_state.orth_loss
            #with open('output.txt', 'w') as file:
                # Redirect the print output to the file
            #    print(enc_state, file=file)
            
            self.svd_plot (data)

    def svd_plot (self, x):
        #self.i=self.i+1
        plt.figure(figsize=(10, 5))
	# compute the singular value decomposition
        u, s, v = torch.svd(x.view(-1,x.size(-1)))
        
        with open('output.txt', 'w') as file:
                # Redirect the print output to the file
                print(x.shape, 'and ', x.view(x.size(0), -1).shape, file=file)
	# compute the normalized cumulative singular values
        cumulative_s = torch.cumsum(s, dim=0) / torch.sum(s)
        normalized_cumulative_s = cumulative_s / cumulative_s[-1]

	# plot the normalized cumulative singular values
        plt.plot(normalized_cumulative_s)
        plt.xlabel('Singular value index')
        plt.ylabel('Normalized cumulative singular value')
        plt.title('Spectral analysis of relations on key space after spreading out through heads')
        plt.savefig("svd_"+f"{self.i}"+".png")

