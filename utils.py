import random
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from typing import Literal
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import progressbar

from hyperparameters import Config



class BAR():

    def __init__(self, length:int):
        self.reset(new_length=length)
    
    def start(self) -> None:
        self.__bar.start()
        self.__k:int = 0
    
    def update(self, increment:int=1, k:int=None, auto_finish:bool=True) -> None:
        if k is None:
            self.__k += increment
        else:
            self.__k = k
        self.__bar.update(self.status())
        if auto_finish and (self.status() >= self.get_finish_line()):
            self.end()
    
    def status(self) -> int:
        return self.__k
    
    def get_finish_line(self) -> int:
        return self.__max_k
    
    def is_done(self) -> bool:
        return self.__is_finished

    def end(self) -> None:
        if not self.is_done():
            self.__bar.finish()
            self.__is_finished = True
    
    def finish(self) -> None:
        self.end()

    def reset(self, new_length:int) -> None:
        self.__bar = progressbar.ProgressBar(maxval=new_length,
                                      widgets=[progressbar.Bar('=', '[', ']'),
                                               ' ',
                                               progressbar.Percentage()
                                              ]
                                     )
        self.__is_finished:bool = False
        self.__k:int = 0
        self.__max_k:int = new_length


    

def plot_process(samples:np.ndarray, labels:list[str]|None=None,
                 save_picture=False, show_plot=True,
                 img_idx=0, img_name:str="plot",
                 folder_path:str=None,
                 title:str=None) -> None:
    '''
    Plots all the dimensions of the generated dataset.
    '''
    if save_picture or show_plot:
        for i in range(samples.shape[1]):
            if labels is not None:
                plt.plot(samples[:,i], label=labels[i])
            else:
                plt.plot(samples[:,i])

        # giving a title to my graph 
        if labels is not None:
            plt.legend()
        if title:
            plt.title(title)
        
        # function to show the plot 
        if save_picture:
            plt.savefig(f"{folder_path}{img_name}-{img_idx}.png")
        if show_plot:
            plt.show()
        plt.clf()


def compare_sequences(real:np.ndarray, fake:np.ndarray,
                      real_label:str="Real sequence", fake_label:str="Fake Sequence",
                      show_graph:bool=False, save_img:bool=False,
                      img_idx:int=0, img_name:str="plot", folder_path:str=None):
    '''
    Plots two graphs with the two sequences.

    Arguments:
        - `real`: the first sequence with dimension [seq_len, data_dim]
        - `fake`: the second sequence with dimension [seq_len, data_dim]
        - `show_graph`: whether to display the graph or not
        - `save_img`: whether to save the image of the graph or not
        - `img_idx`: the id of the graph that will be used to name the file
        - `img_name`: the file name of the graph that will be used to name the file
        - `folder_path`: path to the folder where to save the image

    Returns:
        - numpy matrix with the pixel values for the image
    '''
    mpl.use('Agg')
    fig, (ax0, ax1) = plt.subplots(2, 1, layout='constrained')
    ax0.set_xlabel('Time-Steps')

    for i in range(real.shape[1]):
        ax0.plot(real.cpu()[:,i])
    ax0.set_ylabel(real_label)

    for i in range(fake.shape[1]):
        ax1.plot(fake.cpu()[:,i])
    ax1.set_ylabel(fake_label)

    if show_graph:
        plt.show()
    if save_img:
        plt.savefig(f"{folder_path}{img_name}-{img_idx}.png")


    # return picture as array
    canvas = fig.canvas
    canvas.draw()  # Draw the canvas, cache the renderer
    plt.clf()

    image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    # NOTE: reversed converts (W, H) from get_width_height to (H, W)
    return image_flat.reshape(*reversed(canvas.get_width_height()), 3)  # (H, W, 3)
    

def set_seed(seed=0) -> None:
    np.random.seed(seed)
    random.seed(seed)


def save_timeseries(samples, folder_path:str, file_name="timeseries.csv") -> None:
    '''
    Save the samples as a csv file.
    '''
    # Save it
    df = pd.DataFrame(samples)
    df.to_csv(f"{folder_path}{file_name}", index=False, header=False)


def corr_heatmap(correlation:np.ndarray,
                 show_pic:bool=True,
                 pic_name:str="correlation-heatmap",
                 save_pic:bool=True,
                 pic_folder:str=None
                 ) -> None:
    '''
    Saves a picture of the correlation matrix as a heatmap.
    '''
    plt.figure()
    sns.heatmap(correlation,
                cmap='RdBu',
                annot=False,
                vmin=-1,
                vmax=1
                )
    if save_pic:
        plt.savefig(f"{pic_folder}{pic_name}.png",dpi=300)
    if show_pic:
        plt.show()


def str_to_datetime(data:str,
                    data_format:str="%Y-%m-%d %H:%M:%S"
                    ) -> datetime.datetime:
    '''
    Given a string following the given format into a date and returns the month.

    Arguments:
        - data: the date as a string.
        - data_format: the format string the data must follow for this function to work.
    '''
    return datetime.datetime.strptime(data, data_format)


def PCA_visualization(nominal_data:np.ndarray,
                      anomalous_data:np.ndarray,
                      label_1:str="Normal",
                      label_2:str="Anomalous",
                      title:str="Distribution comparison",
                      show_plot:bool=False,
                      save_plot:bool=True,
                      folder_path:str=None,
                      img_name:str="pca-visual",
                      verbose:bool=False
                      ) -> None:
    """
    Using PCA for generated and original data visualization
     on both the original and synthetic datasets (flattening the temporal dimension).
     This visualizes how closely the distribution of generated samples
     resembles that of the original in 2-dimensional space

    Arguments:
    - `nominal_data`: original data ( num_sequences, data_dim )
    - `anomalous_data`: generated synthetic data ( num_sequences, data_dim )
    - `show_plot`: display the plot
    - `save_plot`: save the .png of the plot
    - `folder_path`: where to save the file
    """  
    if show_plot or save_plot:
        # Data preprocessing
        N1 = nominal_data.shape[0]
        N2 = anomalous_data.shape[0]
        p = nominal_data.shape[1]
        assert(anomalous_data.shape[1] == p)

        prep_data = nominal_data.reshape((N1,p))
        prep_data_hat = anomalous_data.reshape((N2,p))
        
        # Visualization parameter        
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        blue = ["blue" for i in range(N1)]
        red = ["red" for i in range(N2)]
        f, ax = plt.subplots(1)    
        plt.scatter(pca_results[:,0], pca_results[:,1],
                    c=blue, alpha = 0.25, label = label_1)
        plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                    c=red, alpha = 0.25, label = label_2)

        ax.legend()  
        plt.title(title)
        if save_plot:
            plt.savefig(f"{folder_path}{img_name}.png")
        if show_plot:
            plt.show()
        plt.clf()


def delete_file(file_path:str) -> None:
    '''
    Deletes the file. 
    '''
    if type(file_path) == str:
        import os
        os.remove(file_path)


def end_program(code:int=0,
                message:str=None
                ) -> None:
    '''
    Closes the process, prints the message (if any).

    Arguments:
        - `code`: the exit status code.
        - `message`: the message to print 
    '''
    if message:
        print(message)
    import os
    os._exit(status=code)


def get_data(features:list,
             csv_folder_path:str,
             db_api=None,
             patients_list:list=None,
             verbose:bool=False,
             del_nans:bool=False,
             del_sick:bool=True,
             get_sick:bool=False,
             sane_threshold:int=0,
             as_dataframe:bool=False,
             deltas:list[str] = [],
             triage_intensity:Literal['GIALLO', 'ROSSO', 'all']='all',
             triage_type:Literal['CVC', 'FAV', 'CVC-Per', 'CVC-Tem', 'all']='all'
             ) -> np.ndarray | pd.DataFrame:
    '''
    Generates the timeseries for training.
    '''
    assert(triage_intensity in ['GIALLO', 'ROSSO', 'all'])
    assert(triage_type in ['CVC', 'FAV', 'CVC-Per', 'CVC-Tem', 'all'])
    assert(not (del_sick and get_sick)), "Request impossible to comply."
    
    import pandas as pd
    
    if not patients_list:
        if not db_api:
            raise EnvironmentError
        patients_list = sorted(db_api.get_patient_list())
    X = pd.DataFrame(columns=features)
    for patient in patients_list:
        if verbose:
            print(f"Retrieving data for patient {patient} [{patients_list.index(patient)+1}/{len(patients_list)}]")
        patient_dataframe = pd.read_csv(f"{csv_folder_path}{patient}.csv")

        if del_sick:
            fav_sick = patient_dataframe['TRIAGE FAV Totale'] <= sane_threshold
            cvc_1_sick = patient_dataframe['TRIAGE CVC-Per Totale'] <= sane_threshold 
            cvc_2_sick = patient_dataframe['TRIAGE CVC-Tem Totale'] <= sane_threshold
            patient_dataframe = patient_dataframe[fav_sick & cvc_1_sick & cvc_2_sick]
            if len(patient_dataframe) == 0:
                continue
        if get_sick:
            if triage_intensity == 'all':
                if triage_type in ['all', 'FAV']:
                    fav_sick = patient_dataframe['TRIAGE FAV Totale'] > sane_threshold
                else:
                    fav_sick = False
                
                if triage_type in ['all','CVC','CVC-Per']:
                    cvc_1_sick = patient_dataframe['TRIAGE CVC-Per Totale'] > sane_threshold
                else:
                    cvc_1_sick = False
                
                if triage_type in ['all', 'CVC', 'CVC-Tem']:
                    cvc_2_sick = patient_dataframe['TRIAGE CVC-Tem Totale'] > sane_threshold
                else:
                    cvc_2_sick = False
            
            elif triage_intensity == 'ROSSO':
                if triage_type in ['all', 'FAV']:
                    fav_sick = patient_dataframe['TRIAGE FAV ROSSO Totale'] > sane_threshold
                else:
                    fav_sick = False
                
                if triage_type in ['all','CVC','CVC-Per']:
                    cvc_1_sick = patient_dataframe['TRIAGE CVC-Per ROSSO Totale'] > sane_threshold
                else:
                    cvc_1_sick = False
                
                if triage_type in ['all', 'CVC', 'CVC-Tem']:
                    cvc_2_sick = patient_dataframe['TRIAGE CVC-Tem ROSSO Totale'] > sane_threshold
                else:
                    cvc_2_sick = False

            elif triage_intensity == 'GIALLO':
                if triage_type in ['all', 'FAV']:
                    fav_sick = (patient_dataframe['TRIAGE FAV GIALLO Totale'] > sane_threshold) & (patient_dataframe['TRIAGE FAV ROSSO Totale'] <= sane_threshold)
                else:
                    fav_sick = False
                
                if triage_type in ['all','CVC','CVC-Per']:
                    cvc_1_sick = (patient_dataframe['TRIAGE CVC-Per GIALLO Totale'] > sane_threshold) & (patient_dataframe['TRIAGE CVC-Per ROSSO Totale'] <= sane_threshold)
                else:
                    cvc_1_sick = False
                
                if triage_type in ['all', 'CVC', 'CVC-Tem']:
                    cvc_2_sick = (patient_dataframe['TRIAGE CVC-Tem GIALLO Totale'] > sane_threshold) & (patient_dataframe['TRIAGE CVC-Tem ROSSO Totale'] <= sane_threshold)
                else:
                    cvc_2_sick = False

            patient_dataframe = patient_dataframe[fav_sick | cvc_1_sick | cvc_2_sick]
            if len(patient_dataframe) == 0:
                continue
        # print("Selected features:", features)
        # print("Patient features:", set(list(patient_dataframe.columns)).intersection(set(features)))
        X = pd.concat([ X, patient_dataframe[features] ])
        
    # Merging different DataFrames creates duplicate indexes
    X = X.loc[:,~X.columns.duplicated()].copy()
    X.reset_index(inplace=True)
    # Some parameters' NaNs can be put to 0
    special_parameters = ['Score FAV',   'Score CVC',
                          'Cefal/Vomit', 'Cram/IpoPA',
                          'Altro*',      'Score Coag'
                          ]
    for f in features:
        if f in special_parameters:
            X[f] = X[f].fillna(0)

    if del_nans:
        if verbose:
            print("Deleting NaNs ...")
        # Delete all rows with NaNs in the targets or in the features
        X_fixed = pd.DataFrame(columns=features)
        event_idx = 0
        for row_idx in range(len(X)):
            keep_row = True
            for feature in features:
                try:
                    if pd.isna(X.loc[row_idx, feature]):
                        keep_row = False
                        break
                except:
                    if pd.isna(X.loc[row_idx, feature]):
                        keep_row = False
                        break
            if keep_row:
                x_row = dict()
                for feature in features:
                    x_row[feature] = X.loc[row_idx, feature]
                
                
                X_fixed.loc[len(X_fixed.index)] = x_row

                #X_fixed = pd.concat([X_fixed, pd.DataFrame(x_row, columns=list(x_row.keys()))])
                event_idx += 1
        if verbose:
            print(f"Removed {len(X)-len(X_fixed)} rows containing NaNs.")
        X = X_fixed
        del X_fixed
    else:
        # Replace NaNs with the mean of the column they belong
        X.fillna(0, inplace=True)
    X = X[features]
    
    if len(deltas) >= 1:
        for param in deltas:
            pre = f"{param} Pre"
            post = f"{param} Post"
            delta = f"{param} Delta"
            if pre in X.columns.values.tolist() and post in X.columns.values.tolist():
                X[delta] = X[pre] - X[post]
                X = X.drop(columns=[pre, post])
    if verbose:
        print(X)
    if as_dataframe:
        return X
    return X.to_numpy()


def get_columns(columns:list,
                csv_folder_path:str,
                db_api=None,
                patients_list:list=None,
                verbose:bool=False,
                del_nans:bool=True,
                as_dataframe:bool=True,
                deltas:list[str] = [],
             ) -> np.ndarray | pd.DataFrame:
    '''
    Reads the patients' CSVs and returns the selected columns.
    '''
    import pandas as pd
    if not patients_list:
        patients_list = sorted(db_api.get_patient_list())
    X = pd.DataFrame(columns=columns)
    for patient in patients_list:
        if verbose:
            print(f"Retrieving data for patient {patient} [{patients_list.index(patient)+1}/{len(patients_list)}]")
        patient_dataframe = pd.read_csv(f"{csv_folder_path}{patient}.csv")
        X = pd.concat([ X, patient_dataframe[columns] ])
        
    # Merging different DataFrames creates duplicate indexes
    X.reset_index(inplace=True)
    # Some parameters' NaNs can be put to 0
    special_parameters = ['Score FAV',   'Score CVC',
                          'Cefal/Vomit', 'Cram/IpoPA',
                          'Altro*',      'Score Coag',
                          'Eparina'
                         ]
    for f in columns:
        if f in special_parameters:
            X[f] = X[f].fillna(0)

    if del_nans:
        X_fixed = X.dropna()
        if verbose:
            print(f"Removed {len(X)-len(X_fixed)} rows containing NaNs.")
        X = X_fixed
        X.reset_index(inplace=True)
        del X_fixed
    else:
        X.fillna(0, inplace=True)
    X = X[columns]
    
    if len(deltas) >= 1:
        for param in deltas:
            pre = f"{param} Pre"
            post = f"{param} Post"
            delta = f"{param} Delta"
            if pre in columns and post in columns:
                X[delta] = X[pre] - X[post]
                X = X.drop(columns=[pre, post])
    if verbose:
        print(X)
    if as_dataframe:
        return X
    return X.to_numpy()


def get_clinical_timeseries(triage_type:Literal['FAV','CVC','all'],
                            features:list[str],
                            deltas:list[str]=[],
                            as_dataframe:bool=False,
                            ) -> tuple[np.ndarray, np.ndarray] | tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Returns the timeseries of the clinical events
    '''
    from database import clinical_events
    all_events = clinical_events(fill_nans=True).dropna()
    if triage_type == "FAV":
        is_event = all_events["Eventi FAV"] >= 1
        is_not_event = all_events["Eventi FAV"] == 0
    elif triage_type in ['CVC', 'CVC-Per', 'CVC-Tem']:
        is_event = all_events["Eventi CVC"] >= 1
        is_not_event = all_events["Eventi CVC"] == 0
    elif triage_type == 'all':
        is_event = (all_events["Eventi CVC"] >= 1) | (all_events["Eventi FAV"] >= 1)
        is_not_event = (all_events["Eventi CVC"] == 0) & (all_events["Eventi FAV"] == 0)
    else:
        assert(False), f"{triage_type} triage is not supported."
    
    no_events = all_events[is_not_event][features]
    events = all_events[is_event][features]

    if len(deltas) >= 1:
        for param in deltas:
            pre = f"{param} Pre"
            post = f"{param} Post"
            delta = f"{param} Delta"
            if pre in features and post in features:
                no_events[delta] = no_events[pre] - no_events[post]
                no_events = no_events.drop(columns=[pre, post])
                
                events[delta] = events[pre] - events[post]
                events = events.drop(columns=[pre, post])
    if as_dataframe:
        return no_events, events
    return no_events.to_numpy(), events.to_numpy()



def min_max_scaling(value:float, min:float, max:float) -> float:
    '''
    Scales the value in a [0,1] range with respect to min and max values, like so:
        return (value - min) / (max - min)
    '''
    assert(min != max), f"Can't have min ({min}) equal to max ({max})"
    return (value - min) / (max - min)


def show_summary_statistics(actual:np.ndarray, 
                            predicted:np.ndarray,
                            model_name:str='model',
                            labels:list[int]=[0,1],
                            normalize:Literal['true', 'pred', 'all']='true',
                            title:str='Confusion Matrix',
                            use_round:bool=False,
                            save_pic:bool=True,
                            pic_folder:str=None,
                            show_plot:bool=True,
                            verbose:bool=True,
                            get_all_stats:bool=False,
                            ) -> np.ndarray | tuple[np.ndarray, float, float, float]:
    '''
    Computes and displays confusion matrix.

    If `get_all_stats=True` returns the tuple `(confusion_matrix, f1_score, precision, recall)`, otherise it returns only the confusion matrix
    '''
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import recall_score
    cm = confusion_matrix(actual,predicted,normalize=normalize)
    if use_round:
        cm = np.round(cm, 4)
    norm = plt.Normalize(0,100)
    if save_pic or show_plot:
        sns.heatmap(cm * 100, 
                    annot=True,
                    fmt='g', 
                    xticklabels=labels,
                    yticklabels=labels,
                    norm=norm
                    )
        plt.ylabel('Prediction',fontsize=13)
        plt.xlabel('Actual',fontsize=13)
        plt.title(title,fontsize=17)

    if save_pic:
        plt.savefig(f"{pic_folder}{model_name}_confusion.png", dpi=200)
        plt.clf()
    if show_plot:
        plt.show()
        plt.clf()

    f1_val = f1_score(actual, predicted, average=None)
    precision_val = precision_score(actual, predicted, average=None)
    recall = recall_score(actual, predicted, average=None)
    if verbose:
        print("Precision: ", precision_val)
        print("Recall:    ", recall)
        print("F1 score:  ", f1_val)
    if get_all_stats:
        return cm, f1_val, precision_val, recall
    return cm


def save_dtree(dtree, file_path:str) -> None:
    import pickle
    pickle.dump(dtree, open(file_path, 'wb'))


def load_dtree(file_path:str) -> None:
    import pickle
    return pickle.load(open(file_path, 'rb'))


def get_avg_tree_impurity(dtree) -> float:
    n_leaves:int = 0
    tot_imp = 0.0
    
    n_nodes = dtree.tree_.node_count
    children_left = dtree.tree_.children_left
    children_right = dtree.tree_.children_right
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        # If the left and right child of a node is not the same we have a split node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack` so we can loop through them
        if is_split_node:
            # is not leaf
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            # is leaf
            n_leaves += 1
            tot_imp += dtree.tree_.impurity[node_id] # entropy impurity at 'node'   
    return tot_imp/n_leaves # average impurity


def get_n_leaves(dtree) -> float:
    n_leaves:int = 0
    tot_imp = 0.0
    
    n_nodes = dtree.tree_.node_count
    children_left = dtree.tree_.children_left
    children_right = dtree.tree_.children_right
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        # If the left and right child of a node is not the same we have a split node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack` so we can loop through them
        if is_split_node:
            # is not leaf
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            # is leaf
            n_leaves += 1
            tot_imp += dtree.tree_.impurity[node_id] # entropy impurity at 'node'   
    return n_leaves 


def save_json(dictionary:dict, file_path:str) -> None:
    import json
    with open(file_path, 'w') as f:
        json.dump(dictionary, f)

def load_json(file_path:str) -> dict:
    import json
    with open(file_path) as f:
        return json.load(f)


def get_files_in_dir(directory:str) -> list[str]:
    import os
    all_files:list[str] = list()
    for file_obj in os.scandir(path=directory):
        file_obj.name
        if file_obj.is_file():
            all_files.append(file_obj.name)
    return all_files


def discretize(value:float, step:float=Config.discretize_step, place_in_middle:bool=False) -> float:
    '''
    Returns another value in its discretized form, by only allowing a mesh over the real space.
    Mesh granularity depends on the `step` argument.
    '''
    try:
        if np.isnan(value):
            return value
    except TypeError:
        return value
    return int(value/step)*step + (0 if not place_in_middle else step/2)



def get_triage_columns(triage_type:str, triage_intensity:str, get_percentage:bool=True) -> list[str]:
    '''
    Wich column contain the desired triage score
    '''
    if triage_type == 'all' and triage_intensity == 'all':
        if get_percentage:
            triage_column = ["TRIAGE Totale %"]
        else:
            triage_column = ["TRIAGE Totale"]
    elif triage_type == 'FAV' and triage_intensity == 'all':
        if get_percentage:
            triage_column = ["TRIAGE FAV Totale %"]
        else:
            triage_column = ["TRIAGE FAV Totale"]
    elif triage_type == 'CVC-Per' and triage_intensity == 'all':
        if get_percentage:
            triage_column = ["TRIAGE CVC-Per Totale %"]
        else:
            triage_column = ["TRIAGE CVC-Per Totale"]
    elif triage_type == 'CVC-Tem' and triage_intensity == 'all':
        if get_percentage:
            triage_column = ["TRIAGE CVC-Tem Totale %"]
        else:    
            triage_column = ["TRIAGE CVC-Tem Totale"]
    elif triage_type == 'CVC' and triage_intensity == 'all':
        if get_percentage:
            triage_column = ["TRIAGE CVC Totale %"]
        else:
            triage_column = ["TRIAGE CVC Totale"]
    elif triage_type == 'all' and triage_intensity == 'GIALLO':
        if get_percentage:
            triage_column = ["TRIAGE FAV GIALLO Totale %",
                         "TRIAGE CVC-Tem GIALLO Totale %",
                         "TRIAGE CVC-Per GIALLO Totale %"]
        else:    
            triage_column = ["TRIAGE FAV GIALLO Totale",
                         "TRIAGE CVC-Tem GIALLO Totale",
                         "TRIAGE CVC-Per GIALLO Totale"]
    elif triage_type == 'FAV' and triage_intensity == 'GIALLO':
        if get_percentage:
            triage_column = ["TRIAGE FAV GIALLO Totale %"]
        else:
            triage_column = ["TRIAGE FAV GIALLO Totale"]
    elif triage_type == 'CVC-Per' and triage_intensity == 'GIALLO':
        if get_percentage:
            triage_column = ["TRIAGE CVC-Per GIALLO Totale %"]
        else:
            triage_column = ["TRIAGE CVC-Per GIALLO Totale"]
    elif triage_type == 'CVC-Tem' and triage_intensity == 'GIALLO':
        if get_percentage:
            triage_column = ["TRIAGE CVC-Tem GIALLO Totale %"]
        else:
            triage_column = ["TRIAGE CVC-Tem GIALLO Totale"]
    elif triage_type == 'CVC' and triage_intensity == 'GIALLO':
        if get_percentage:
            triage_column = ["TRIAGE CVC-Tem GIALLO Totale %",
                         "TRIAGE CVC-Per GIALLO Totale %"
                         ]
        else:
            triage_column = ["TRIAGE CVC-Tem GIALLO Totale",
                         "TRIAGE CVC-Per GIALLO Totale"
                         ]
    elif triage_type == 'all' and triage_intensity == 'ROSSO':
        if get_percentage:
            triage_column = ["TRIAGE FAV ROSSO Totale %",
                         "TRIAGE CVC-Tem ROSSO Totale %",
                         "TRIAGE CVC-Per ROSSO Totale %"]
        else:
            triage_column = ["TRIAGE FAV ROSSO Totale",
                         "TRIAGE CVC-Tem ROSSO Totale",
                         "TRIAGE CVC-Per ROSSO Totale"]
    elif triage_type == 'FAV' and triage_intensity == 'ROSSO':
        if get_percentage:
            triage_column = ["TRIAGE FAV ROSSO Totale %"]
        else:
            triage_column = ["TRIAGE FAV ROSSO Totale"]
    elif triage_type == 'CVC-Per' and triage_intensity == 'ROSSO':
        if get_percentage:
            triage_column = ["TRIAGE CVC-Per ROSSO Totale %"]
        else:
            triage_column = ["TRIAGE CVC-Per ROSSO Totale"]
    elif triage_type == 'CVC-Tem' and triage_intensity == 'ROSSO':
        if get_percentage:
            triage_column = ["TRIAGE CVC-Tem ROSSO Totale %"]
        else:
            triage_column = ["TRIAGE CVC-Tem ROSSO Totale"]
    elif triage_type == 'CVC' and triage_intensity == 'ROSSO':
        if get_percentage:
            triage_column = ["TRIAGE CVC-Tem ROSSO Totale %",
                         "TRIAGE CVC-Per ROSSO Totale %"
                         ]
        else:
            triage_column = ["TRIAGE CVC-Tem ROSSO Totale",
                         "TRIAGE CVC-Per ROSSO Totale"
                         ]
    else:
        raise ValueError
    return triage_column


import time 
class TimeExecution():
    
    def __init__(self):
        self.__start_time = None
        self.__end_time = None
    
    def elapsed(self):
        assert(not (self.__end_time is None))
        assert(not (self.__start_time is None))
        return self.__end_time - self.__start_time
    
    def start(self) -> None:
        self.__start_time = time.time()
        self.__end_time = None
    
    def end(self) -> None:
        assert(not (self.__start_time is None))
        self.__end_time = time.time()

    def print_time(self, silent:bool=False, digits:int=2) -> str:
        '''
        If `silent` it returns the string without printing it.
        '''
        seconds_total = self.elapsed()
        minutes:int = int(seconds_total // 60)
        hours:int = int(minutes // 60)
        minutes = int(minutes % 60)
        seconds:int = round(seconds_total % 60, digits)
        string = f"{hours}h {minutes}m {seconds}s"
        if not silent:
            print(string)
        else:
            return string



def get_data_avg(features:list,
                 csv_folder_path:str,
                 db_api=None,
                 patients_list:list=None,
                 verbose:bool=False,
                 del_nans:bool=False,
                 del_sick:bool=True,
                 get_sick:bool=False,
                 sane_threshold:int=0,
                 as_dataframe:bool=False,
                 deltas:list[str] = [],
                 triage_intensity:Literal['GIALLO', 'ROSSO', 'all']='all',
                 triage_type:Literal['CVC', 'FAV', 'CVC-Per', 'CVC-Tem', 'all']='all',
                 lookback:int=31,
                ) -> np.ndarray | pd.DataFrame:
    '''
    Generates the timeseries for training.
    '''
    assert lookback > 0
    X:pd.DataFrame = get_data(features=features,
                              csv_folder_path=csv_folder_path,
                              db_api=db_api,
                              patients_list=patients_list,
                              verbose=verbose,
                              del_nans=del_nans,
                              del_sick=del_sick,
                              get_sick=get_sick,
                              sane_threshold=sane_threshold,
                              as_dataframe=True,
                              deltas=deltas,
                              triage_intensity=triage_intensity,
                              triage_type=triage_type
                             )
    Y:pd.DataFrame = pd.DataFrame(columns=list(X.columns))

    for i, _ in X.iterrows():
        aggregate_row:dict[str,float|int] = dict()
        for column in X.columns:
            aggregate_row[column] = [0]
        k:int = 1
        for offset in range(lookback):
            try:
                row = X.iloc[i+offset]
                for column in X.columns:
                    aggregate_row[column][0] += row[column]
                k += 1
            except IndexError:
                break
        for column in X.columns:
            aggregate_row[column][0] /= k
        Y = pd.concat([Y, pd.DataFrame(aggregate_row)])

    if as_dataframe:
        return X
    else:
        return X.to_numpy()
    


def get_dataframe(file_path:str, sheet_name:str=None, date_column:str=None, date_format:str="%Y-%m-%d %H:%M:%S") -> pd.DataFrame:
    import pandas as pd
    try:
        xl = pd.ExcelFile(file_path)
        DF = xl.parse(sheet_name=sheet_name)
    except ValueError:
        if file_path.endswith(".csv"):
            DF = pd.read_csv(file_path)
        elif file_path.endswith(".tsv"):
            DF = pd.read_csv(file_path, sep='\t')
        else:
            DF = TypeError(f"Unsupported file extension for file '{file_path}'")
    if not (date_column is None):
        DF[date_column] = [datetime.datetime.strptime(d, date_format) for d in DF[date_column]]

    return DF
    


def scan_dir(folder_path:str) -> tuple[list[str], list[str]]:
    '''
    Returns tuple (`files_found`, `folders_found`)
    '''
    import os
    folders_found:list[str] = list()
    files_found:list[str] = list()

    for obj in os.scandir(path=folder_path):
        if obj.is_dir():
            folders_found.append(obj.name)
        elif obj.is_file():
            files_found.append(obj.name)
    return files_found, folders_found


def print_colored(text, color:str, highlight:str=None, end:str='\n') -> None:
    '''
    Supported colors:
    - black	
    - red	
    - green	
    - yellow	
    - blue	
    - magenta	
    - cyan	
    - white	
    - light_grey	
    - dark_grey	
    - light_red	
    - light_green	
    - light_yellow	
    - light_blue	
    - light_magenta	
    - light_cyan

    Highlighted text:
    - on_{color}
    '''
    from termcolor import colored
    print(colored(str(text), color, highlight), end=end)



def months_between(date1:datetime.datetime, date2:datetime.datetime) -> int:
    # Ensure date1 is the earlier date
    if date1 > date2:
        date1, date2 = date2, date1
    
    # Calculate year and month difference
    months = (date2.year - date1.year) * 12 + date2.month - date1.month
    
    # If day of date2 is less than day of date1, subtract one month
    if date2.day < date1.day:
        months -= 1

    return int(months)



def years_between(start, end) -> int:
    return int((end - start).days / 365.25)  # Approximate, accounts for leap years


from concurrent import futures
def embarassing_parallelism(function, n_workers:int, arguments_list:list, use_process:bool=True) -> list:
    '''
    Launches `n_workers` parallel jobs, with no communication between them _(embarassing parallelisms)_ and returns the **unordered** iterable list with their results.
    Each job `i` executes `function` with input `arguments_list[i]`, we recommend the input to be a dictionary.
    
    Returns the iterable `results` with each job's results, we recommend the iterable to return a dictionary.

    ### Arguments ###
    - `function`: the executable to be ran
    - `n_workers`: number of parallel jobs to launch
    - `arguments_list`: list of inputs for the jobs _(see above description)_
    - `use_process`: if `True` it launches `n_workers` **processes**, else it launches `n_workers` **threads**
    '''
    if len(arguments_list) != n_workers:
        raise IndexError(f"Length of 'argument_list' (now {len(arguments_list)}) must be equal to 'n_workers' (now {n_workers}).")
    if use_process:
        with futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(function,
                                                   arguments_list,
                                                  )
                                     )
    else:
        with futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(function,
                                                   arguments_list,
                                                  )
                                     )
    return results


def dict_to_dataframe_row(data:dict) -> pd.DataFrame:
    new_data:dict = dict()
    for k,v in data.items():
        new_data[k] = [v]
    return pd.DataFrame(new_data)