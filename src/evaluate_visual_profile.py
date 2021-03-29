import argparse
import pandas as pd
import numpy as np
import os
import csv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageOps


def parse_args():
    parser = argparse.ArgumentParser(description="Run explanations evaluation.")
    parser.add_argument('--dataset', type=str, default='amazon_boys_girls_reduced', help='dataset name')
    parser.add_argument('--image_feat_extractors', nargs='+', type=str, default=['alexnet',
                                                                                 'resnet50',
                                                                                 'vgg19'],
                        help='image feature extractors')
    parser.add_argument('--visual_recommenders', nargs='+', type=str,
                        default=['VBPR', 'DeepStyle', 'ACF', 'VNPR'], help='visual recommenders')
    parser.add_argument('--top_k', type=int, default=100, help='top-k to retrieve')
    parser.add_argument('--save_plots', type=bool, default=False, help='whether to save or not plots')

    return parser.parse_args()


def evaluate_visual_profile():
    feats_dict = {
        'alexnet': {
            'VBPR': 'cnn_features_alexnet_5.npy',
            'DeepStyle': 'cnn_features_alexnet_5.npy',
            'ACF': 'cnn_features_alexnet_0.npy',
            'VNPR': 'category_features_alexnet_5_pca128.npy'
        },
        'resnet50': {
            'VBPR': 'cnn_features_resnet50_avg_pool.npy',
            'DeepStyle': 'cnn_features_resnet50_avg_pool.npy',
            'ACF': 'cnn_features_resnet50_conv5_block3_out.npy',
            'VNPR': 'category_features_resnet50_avg_pool_pca128.npy'
        },
        'vgg19': {
            'VBPR': 'cnn_features_vgg19_fc2.npy',
            'DeepStyle': 'cnn_features_vgg19_fc2.npy',
            'ACF': 'cnn_features_vgg19_block5_pool.npy',
            'VNPR': 'category_features_vgg19_fc2_pca128.npy'
        }
    }

    args = parse_args()

    training = pd.read_csv('../data/{0}/trainingset.tsv'.format(args.dataset), sep='\t', header=None)
    num_users = training[0].nunique()

    for visual_rec in args.visual_recommenders:
        for cnn in args.image_feat_extractors:
            results = pd.read_csv('../results/{0}_{1}/{2}.tsv'.format(args.dataset, cnn, visual_rec), sep='\t', header=None)

            features = np.load('../data/{0}_{1}/original/{2}'.format(args.dataset, cnn, feats_dict[cnn][visual_rec]))
            if visual_rec == 'ACF':
                features = features.reshape((features.shape[0], features.shape[1] * features.shape[2]))

            if not os.path.exists('../plots/{0}_{1}/{2}/{3}/'.format(args.dataset, args.top_k, visual_rec, cnn)):
                os.makedirs('../plots/{0}_{1}/{2}/{3}/'.format(args.dataset, args.top_k, visual_rec, cnn))

            distance = 0.0
            max_dist = - np.inf
            max_dist_u = 0
            min_dist = + np.inf
            min_dist_u = 0
            all_distances = []

            with open('../plots/{0}_{1}/{2}/{3}/all_users_stats.csv'.format(args.dataset, args.top_k, visual_rec, cnn), 'w') as c:
                fieldnames = ['User', 'Distance']
                writer = csv.DictWriter(c, fieldnames=fieldnames)
                writer.writeheader()
                for u in range(num_users):
                    pos = training[training[0] == u][1].to_list()
                    top_k = results[results[0] == u].head(args.top_k)[1].to_list()

                    # load ground-truth user positive features for a specific cnn and a specific visual recommender
                    pos_features = features[pos]

                    # load predicted user features for top-k for a specific cnn and a specific visual recommender
                    top_k_features = features[top_k]

                    t_sne = TSNE(n_components=2, random_state=1234).fit_transform(np.concatenate((pos_features, top_k_features), axis=0))

                    # find centroids of ground-truth and predicted
                    pos_centroid_x = np.mean(t_sne[:len(pos_features), 0])
                    pos_centroid_y = np.mean(t_sne[:len(pos_features), 1])
                    pos_centroid = np.array([pos_centroid_x, pos_centroid_y])
                    top_k_centroid_x = np.mean(t_sne[len(pos_features):, 0])
                    top_k_centroid_y = np.mean(t_sne[len(pos_features):, 1])
                    top_k_centroid = np.array([top_k_centroid_x, top_k_centroid_y])
                    current_dist = np.linalg.norm(pos_centroid - top_k_centroid)

                    writer.writerow({
                        'User': u,
                        'Distance': current_dist
                    })

                    all_distances.append(current_dist)
                    distance += current_dist

                    if current_dist < min_dist:
                        min_dist = current_dist
                        min_dist_u = u

                    if current_dist > max_dist:
                        max_dist = current_dist
                        max_dist_u = u

                    if args.save_plots:
                        fig, ax = plt.subplots()

                        # positive items
                        artists = []
                        for idx, (x, y) in enumerate(zip(t_sne[:len(pos_features), 0], t_sne[:len(pos_features), 1])):
                            im = Image.open('../data/{0}_{1}/original/images/{2}.jpg'.format(args.dataset, cnn, str(pos[idx])))
                            im = ImageOps.expand(im, border=20, fill='green')
                            im = OffsetImage(im, zoom=0.1)
                            ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
                            artists.append(ax.add_artist(ab))
                        ax.update_datalim(np.column_stack([t_sne[:len(pos_features), 0], t_sne[:len(pos_features), 1]]))
                        ax.autoscale()

                        # predicted items
                        artists = []
                        for idx, (x, y) in enumerate(zip(t_sne[len(pos_features):, 0], t_sne[len(pos_features):, 1])):
                            im = Image.open('../data/{0}_{1}/original/images/{2}.jpg'.format(args.dataset, cnn, str(top_k[idx])))
                            im = ImageOps.expand(im, border=20, fill='red')
                            im = OffsetImage(im, zoom=0.1)
                            ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
                            artists.append(ax.add_artist(ab))
                        ax.update_datalim(np.column_stack([t_sne[len(pos_features):, 0], t_sne[len(pos_features):, 1]]))
                        ax.autoscale()

                        # plt.scatter(t_sne[:len(pos_features), 0], t_sne[:len(pos_features), 1], marker='*')
                        plt.scatter(pos_centroid[0], pos_centroid[1], marker="X", edgecolors='green')
                        plt.scatter(top_k_centroid[0], top_k_centroid[1], marker="X", edgecolors='red')
                        # frame = plt.plot([pos_centroid[0], top_k_centroid[0]], [pos_centroid[1], top_k_centroid[1]], color='black')
                        # frame = plt.scatter(t_sne[len(pos_features):, 0], t_sne[len(pos_features):, 1])
                        plt.plot([pos_centroid[0], top_k_centroid[0]], [pos_centroid[1], top_k_centroid[1]], color='black')
                        # frame.axes.get_xaxis().set_visible(False)
                        # frame.axes.get_yaxis().set_visible(False)
                        plt.axis('off')
                        plt.savefig('../plots/{0}_{1}/{2}/{3}/u_{4}.pdf'.format(args.dataset, args.top_k, visual_rec, cnn, u), bbox_inches='tight')
                        plt.close()

            std_dev = np.std(np.array(all_distances))
            with open('../plots/{0}_{1}/{2}/{3}/final_stats.out'.format(args.dataset, args.top_k, visual_rec, cnn), 'w') as f:
                f.write('Average distance: {0}\n'.format(distance / num_users))
                f.write('Standard deviation: {0}\n'.format(std_dev))
                f.write('User with max distance: {0}, distance: {1}\n'.format(max_dist_u, max_dist))
                f.write('User with min distance: {0}, distance: {1}\n'.format(min_dist_u, min_dist))


if __name__ == '__main__':
    evaluate_visual_profile()
