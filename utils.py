import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import base
from scipy import spatial
from time import time
        
class SubsampleTrainCV():
    """ Splits train and validations sets by following:
        - random sample of test set, number of observations is defined as int(test_ratio*n), 
            where n is size of data set
        - from remaining observations, sample the observations for train set, 
            number of observations is defined as int((n-test_size)*train_ratio)
        the sampling is performed independently n_splits times
        
        Note: actually it can be made simpler with two 'train_test_split' calls
    """
    def __init__(self, n, n_splits=3, test_ratio=1/3, train_ratio=0.7, random_state=19999):
        self.n=n
        self.n_splits=n_splits
        self.test_ratio=test_ratio
        self.train_ratio=train_ratio
        self.random_state=random_state
        
        all_ind=range(n)
        self.test_size=int(n*test_ratio)
        self.train_size=int((n-int(n*test_ratio))*self.train_ratio)
        
        test_indices=[]
        train_indices=[]
        
        # for reproducibility
        np.random.seed(self.random_state)
        
        for i in range(self.n_splits):
            test_indice=np.random.choice(all_ind, size=self.test_size, replace=False)
            test_indices.append(test_indice)
            train_indice_0=list(set(all_ind).difference(set(test_indice)))
            train_indice=np.random.choice(train_indice_0, size=self.train_size, replace=False)
            train_indices.append(train_indice)
                
        self.train_indices=train_indices
        self.test_indices=test_indices
        
    def __str__(self):
        return 'subs{:.1f}CV'.format(self.train_ratio)
    def split(self, X, y=None, groups=None):
        for tr,ts in zip(self.train_indices, self.test_indices):
            yield tr,ts
            
    def get_n_splits(self, X, y=None, groups=None):
        return self.n_splits
    
    def draw_map(self, coords, figsize_wid=10):
        fig, axes=plt.subplots(nrows=1, ncols=self.n_splits, figsize=(figsize_wid*self.n_splits, figsize_wid))
        
        for i in range(self.n_splits):
            tr=self.train_indices[i]
            ts=self.test_indices[i]
            trcrds=coords.iloc[tr]
            tscrds=coords.iloc[ts]
            axes[i].set_title('train_ratio={:.2f}'.format(self.train_ratio), fontsize=30)
            ms=9
            axes[i].scatter(tscrds['x1'], tscrds['x2'], marker='.', c='grey', label='test', s=ms)
            axes[i].scatter(trcrds['x1'], trcrds['x2'], marker='.', c='seagreen', label='train', s=ms)
            lgnd=axes[i].legend(prop={'size': 30})
            for lh in lgnd.legendHandles:
                lh._sizes=[400]
        plt.tight_layout()
        plt.show()

class LandRectanglesCV():
    """ Split train and test by splitting map to rectangles
        and selecting some of rects for train and rest for test
    """
    def __init__(self, coords, n_splits_by_axis=10, test_ratio=1/3, n_splits=3, random_state=19999):
        self.n_splits_by_axis=n_splits_by_axis
        self.test_ratio=test_ratio
        self.n_splits=n_splits
        self.random_state=random_state
    
        coords_min=coords.min(axis=0)
        coords_max=coords.max(axis=0)
                
        rect_indices=pd.Series([(i,j) for i in range(n_splits_by_axis) for j in range(n_splits_by_axis)])
        
        #for reproducibility
        np.random.seed(random_state)

        train_rect_indices = [rect_indices.sample(n=int((1-self.test_ratio) * n_splits_by_axis**2), replace=False) for j in range(n_splits)]
        coords_to_rects=np.floor((coords-coords_min)/(coords_max-coords_min)*n_splits_by_axis).astype(int)
        coords_tuples=coords_to_rects.apply(lambda x: tuple(x), axis=1)
        train_masks=[coords_tuples.isin(train_rect_index) for train_rect_index in train_rect_indices]
        
        self.coords=coords
        self.train_indices=[coords[train_mask].index for train_mask in train_masks]
        self.test_indices=[coords[~train_mask].index for train_mask in train_masks]
    
    def __str__(self):
        return 'rectCV'
        
    def split(self, X, y=None, groups=None):
        for tr,ts in zip(self.train_indices, self.test_indices):
            yield tr,ts
            
    def get_n_splits(self, X, y=None, groups=None):
        return self.n_splits
    
    def draw_map(self, save_name='land_rectangles_cv.pdf', figsize_wid=3):
        fig, axes=plt.subplots(nrows=1, ncols=self.n_splits, figsize=(figsize_wid*self.n_splits, figsize_wid))
        fs=int(30/10*figsize_wid)
        
        for i in range(self.n_splits):
            tr=self.train_indices[i]
            ts=self.test_indices[i]
            trcrds=self.coords.iloc[tr]
            tscrds=self.coords.iloc[ts]
            ms=9
            axes[i].scatter(tscrds['x1'], tscrds['x2'], marker='.', c='gray', label='test', s=ms, rasterized=True)
            axes[i].scatter(trcrds['x1'], trcrds['x2'], marker='.', c='seagreen', label='train', s=ms, rasterized=True)
            lgnd=axes[i].legend(prop={'size': fs})
            for lh in lgnd.legendHandles:
                lh._sizes=[int(400/10*figsize_wid)]
        plt.tight_layout()
        plt.show()

class TimedCluster():
    """
        Transform the data with shape (n, n_features*n_periods) to clusters
        
        Parameters:
        ----------
        
        cluster_model: base unsupervised cluster model
        feature_space: 'features' or 'periods', defines space for cluster model
            for 'features' every cluster model will receive data with shape (n, n_periods), 
                total n_features models will be trained
            for 'periods' every cluster model will receive data with shape (n, n_features), 
                total n_periods models will be trained, also in this case data should be normalized before
        n_features: number of features in one period
        n_periods: number of snapshots in data
    """
    def __init__(self, cluster_model, feature_space='features', n_features=10, n_periods=23):
        self.cluster_model=cluster_model
        self.feature_space=feature_space
        self.n_features=n_features
        self.n_periods=n_periods
        
    def fit(self, X, y=None):
        cluster_models=[]
        if self.feature_space=='features':
            for f in range(self.n_features):
                cls=base.clone(self.cluster_model)
                cls.fit(X.iloc[:,f::self.n_features])
                cluster_models.append(cls)
        elif self.feature_space=='periods':
            for p in range(self.n_periods):
                cls=base.clone(self.cluster_model)
                cls.fit(X.iloc[:,p*self.n_features:(p+1)*self.n_features])
                cluster_models.append(cls)
        else:
            raise AttributeError()
        self.cluster_models=cluster_models
        return self
        
    def predict(self, X):
        res=[]
        if self.feature_space=='features':
            for f in range(self.n_features):
                cls=self.cluster_models[f]
                clspred=cls.predict(X.iloc[:,f::self.n_features])
                res.append(clspred)
        elif self.feature_space=='periods':
            for p in range(self.n_periods):
                cls=self.cluster_models[p]
                clspred=cls.predict(X.iloc[:,p*self.n_features:(p+1)*self.n_features])
                res.append(clspred)
        else:
            raise AttributeError()

        res=np.column_stack(res)
        return res

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)
        
class NearestPointsEmbedding():
    """
        Embed observation with features from 'n_neighbors' nearest points
    """
    def __init__(self, n_neighbors=7):
        self.n_neighbors=n_neighbors
        
    def fit(self, X, y=None):
        self.train=X.copy()
        self.train_coord_set=set(X[['x1','x2']].apply(lambda x: tuple(x), axis=1))
        return self
        
    def transform(self, X):
        data_coord_set=set(X[['x1','x2']].apply(lambda x: tuple(x), axis=1))
        
        # this is to avoid duplication in case of fit and transform on same set
        # in case of different fit and trasnform data, we combine them, otherwise use only train
        if data_coord_set.difference(self.train_coord_set):
            train_ext=self.train.append(X)
        else:
            train_ext=self.train

        self.kd=spatial.KDTree(train_ext[['x1','x2']], leafsize=10)
        
        local_area_ind=self.kd.query(X[['x1','x2']].values, k=self.n_neighbors)[1]
        
        train_ext=train_ext.drop(['x1','x2'], axis=1)
        
        # create result, the first neighbor is the point itself
        res=np.hstack( train_ext.iloc[local_area_ind[:,i]].reset_index(drop=True).values.astype(np.int16)
                        for i in range(self.n_neighbors)
                     )
        
        return res

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

def test_cv(X, y, clf, cv, score, description):
    start=time()
    
    res=pd.DataFrame(columns=['clf', 'description', 'cv', 'score_mean'])
    scores=[]
    
    for train_index, test_index in cv.split(X, y):
        if isinstance(X, pd.DataFrame):
            _X_train, _X_test = X.iloc[train_index], X.iloc[test_index]
        else:
            _X_train, _X_test = X[train_index], X[test_index]
        _y_train, _y_test = y[train_index], y[test_index]
        
        y_pred = clf.fit(_X_train, _y_train).predict(_X_test)
        scores.append(score(y_pred, _y_test))
        
    res.loc[0]=[str(clf), description, str(cv), np.round(np.mean(scores), decimals=4)]
    
    print('time: {:.2f}s'.format(time()-start))
    return res
    
# Here the cross validation is trickier: we should transform dataset after split, not before as in regular 
def test_cv_transform(X, y, clf, cv, score, description, transformer):
    start=time()
    
    res=pd.DataFrame(columns=['clf', 'description', 'cv', 'score_mean'])
    scores=[]
    
    for train_index, test_index in cv.split(X, y):
        if isinstance(X, pd.DataFrame):
            _X_train, _X_test = X.iloc[train_index], X.iloc[test_index]
        else:
            _X_train, _X_test = X[train_index], X[test_index]
        _y_train, _y_test = y[train_index], y[test_index]
        
        # inner cv transform:
        _X_train=transformer.fit_transform(_X_train)
        _X_test=transformer.transform(_X_test)        
        
        y_pred = clf.fit(_X_train, _y_train).predict(_X_test)
        scores.append(score(y_pred, _y_test))
        
    res.loc[0]=[str(clf), description, str(cv), np.round(np.mean(scores), decimals=4)]
    
    print('time: {:.2f}s'.format(time()-start))
    return res