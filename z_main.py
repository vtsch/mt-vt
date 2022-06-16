import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from z_dataloader import loadtwoclusters, loadfiveclusters, create_irregular_ts, create_pandas_df, reshape_lstm
from z_clustering_algorithms import sklearnkmeans, k_means_dtw
from z_utils import plot_centroids, plot_umap, plot_loss
from z_embeddings import umap_embedding, lstm_embedding, simple_autoencoder, deep_autoencoder, lstm_autoencoder, bilstm_autoencoder


def kmeans_and_umap(embedding, n_clusters, y, name):
    centroids, kmeans_labels = sklearnkmeans(embedding, n_clusters)
    plot_centroids(centroids, n_clusters, "kmeans cluster centers %s" %name)

    centroids, dtwkmeans_labels = k_means_dtw(embedding, n_clusters, num_iter=5,w=5)
    plot_centroids(centroids, n_clusters, "dtw kmeans cluster centers %s" %name)

    plot_umap(embedding, kmeans_labels, "%s with kmeans labels" %name)
    plot_umap(embedding, dtwkmeans_labels, "%s embeddings with dtw kmeans labels" %name)

    print("ARI kmeans %s : %d" %(name, adjusted_rand_score(y, kmeans_labels)))
    print("ARI dtw kmeans %s : %d" %(name, adjusted_rand_score(y, dtwkmeans_labels)))




if __name__ == '__main__':

    file_name_train = 'data/mitbih_train.csv'
    file_name_test = 'data/mitbih_test.csv'
    n_clusters = 2

    # booleans
    ORIG_DATA = False
    UMAP_EMBEDDINGS = False
    LSTM_EMBEDDINGS = False
    SIMPLE_AC_EMBEDDINGS = False
    DEEP_AC_EMBEDDINGS = False
    LSTM_AC_EMBEDDINGS = False
    BILSTM_AC_EMBEDDINGS = True


    if n_clusters == 2:
        X_train, X_test, y_train, y_test = loadtwoclusters(file_name_train, file_name_test)
    elif n_clusters == 5:
        X_train, X_test, y_train, y_test = loadfiveclusters(file_name_train, file_name_test)

    #check number of zeros in each row in original data (already irregular)
    #n_zeros = np.count_nonzero(X_train==0, axis=1)
    #print("n_zeros: ", n_zeros)

    X_train = create_irregular_ts(X_train)
    X_test = create_irregular_ts(X_test)

    umap_emb = umap_embedding(X_train)

    train, test = create_pandas_df(X_train, X_test)

    # plot centroids of kmeans and dtw kmeans on raw data

    if ORIG_DATA == True:
        centroids, kmeans_labels = sklearnkmeans(X_train, n_clusters)
        plot_centroids(centroids, n_clusters, "kmeans centroids original data")

        centroids, dtwkmeans_labels = k_means_dtw(X_train, n_clusters, num_iter=5,w=5)
        plot_centroids(centroids, n_clusters, "dtw kmeans centroids original data")

        plot_umap(umap_emb, y_train, "umap embedding original data")

    if UMAP_EMBEDDINGS == True:
        kmeans_and_umap(umap_emb, n_clusters, y_train, name="umap embeddings")
    
    if LSTM_EMBEDDINGS == True:
        n_features = 1
        timesteps = 187
        X_train, X_test = reshape_lstm(X_train, X_test, timesteps, n_features)
        model = lstm_embedding(timesteps, n_features)

        history_lstm_seq = model.fit(X_train, X_train, epochs=10,
          batch_size=16,
          shuffle=True,
          validation_data=(X_test, X_test))
        plot_loss(history_lstm_seq, "LSTM loss")

    if SIMPLE_AC_EMBEDDINGS == True:
        model, encoded, encoder_ac = simple_autoencoder()
        history_ac = model.fit(train, train,
                epochs=500,
                batch_size=16,
                shuffle=True,
                validation_data=(test, test))
        
        plot_loss(history_ac, "Simple autoencoder loss")
        print("start predicting")
        encoded_data_ac = encoder_ac.predict(test)
        print("encoded data shape: ", encoded_data_ac.shape)
        print("start kmeans and umap")
        kmeans_and_umap(encoded_data_ac, n_clusters, y_test, name="Simple autoencoder embeddings")

    if DEEP_AC_EMBEDDINGS == True:
        model, encoded, encoder_ac_deep = deep_autoencoder()
        history_ac_deep = model.fit(train, train,
                epochs=500,
                batch_size=16,
                shuffle=True,
                validation_data=(test, test))
        
        plot_loss(history_ac_deep, "Deep autoencoder loss")
        print("start predicting")
        encoded_data_ac_deep = encoder_ac_deep.predict(test)
        print("encoded data shape: ", encoded_data_ac_deep.shape)
        print("start kmeans and umap")
        kmeans_and_umap(encoded_data_ac_deep, n_clusters, y_test, name="Deep autoencoder embeddings")

    if LSTM_AC_EMBEDDINGS == True:
        n_features = 1
        timesteps = 187

        X_train, X_test = reshape_lstm(X_train, X_test, timesteps, n_features)

        lstm_autoencoder = lstm_autoencoder(timesteps)
        lstm_ac_history = lstm_autoencoder.fit(X_train, X_train, 
                                                epochs=5, 
                                                batch_size=16, 
                                                validation_data=(X_test, X_test),
                                                verbose=2)


        plot_loss(lstm_ac_history, "LSTM loss")

        print("start predicting")
        encoded_data_ac_lstm_orig = lstm_autoencoder.predict(X_test)
        print(encoded_data_ac_lstm_orig.shape)
        encoded_data_ac_lstm = encoded_data_ac_lstm_orig.reshape(-1, timesteps)
        print("encoded data shape: ", encoded_data_ac_lstm.shape)
        print("start kmeans and umap")
        kmeans_and_umap(encoded_data_ac_lstm, n_clusters, y_test, name="LSTM autoencoder embeddings")

    if BILSTM_AC_EMBEDDINGS == True:
        n_features = 1
        timesteps = 187

        #X_train, X_test = reshape_lstm(X_train, X_test, timesteps, n_features)

        bilstm_autoencoder = bilstm_autoencoder(timesteps)
        bilstm_ac_history = bilstm_autoencoder.fit(X_train, X_train, 
                                                epochs=5, 
                                                batch_size=16, 
                                                validation_data=(X_test, X_test),
                                                verbose=2)


        plot_loss(lstm_ac_history, "BiLSTM loss")

        print("start predicting")
        encoded_data_bilstm = bilstm_autoencoder.predict(X_test)
        print("start kmeans and umap")
        kmeans_and_umap(encoded_data_bilstm, n_clusters, y_test, name="BiLSTM autoencoder embeddings")



    





