#Find close words based on pretrained language model embedding space
import sys
sys.path.insert(0, '../../python/')
import caffe
import copy
import pickle as pkl
import numpy as np

caffe.set_mode_gpu()
caffe.set_device(2)

vocab_file = '../coco_caption/h5_data/buffer_100/vocabulary.txt'
vocab_lines = open(vocab_file, 'r').readlines()
vocab_lines = [v.strip() for v in vocab_lines]
vocab_lines = ['EOS'] + vocab_lines

attributes = pkl.load(open('../coco_attribute/attribute_lists/attributes_JJ100_NN300_VB100.pkl','rb'))

def get_embedding_vectors():
  model_pretrain_lm = 'mrnn.lm.direct_deploy.prototxt'
  model_pretrain_lm_weights = '/x/lisaanne/mrnn/snapshots_final/mrnn.direct_iter_110000.caffemodel'
  
  lm_net = caffe.Net(model_pretrain_lm, model_pretrain_lm_weights, caffe.TRAIN)
  
  lm_net.blobs['cont_sentence'].reshape(1,len(vocab_lines))
  lm_net.blobs['input_sentence'].reshape(1,len(vocab_lines))
  
  cont_sentence = np.zeros((1, len(vocab_lines)))
  input_sentence = np.zeros((1, len(vocab_lines)))
  input_sentence[0,:] = range(len(vocab_lines))
  
  lm_net.forward(cont_sentence=cont_sentence, input_sentence=input_sentence)
  return copy.deepcopy(lm_net.blobs['embedded_input_sentence'].data)

def cosine_matrix(vector, mat):
    numerator = np.dot(vector, mat.T)
    denominator = np.linalg.norm(vector)*np.linalg.norm(mat, axis=1)
    return (numerator/denominator)

def find_close_words(word, illegal_words,num_words=1):
  restrict_close_words = 1000
  word_idx = vocab_lines.index(word)
  embedding_vectors = get_embedding_vectors().squeeze(0)
  word_vector = embedding_vectors[word_idx,:]
  word_sims = cosine_matrix(word_vector, embedding_vectors)
  for iw in illegal_words:
    word_sims[vocab_lines.index(iw)] = -10000  #so illgal words won't be close
  word_sims = word_sims[:restrict_close_words]

  similar_words =  [vocab_lines[w] for w in np.argsort(word_sims)[-num_words:]]
  sim_scores = np.sort(word_sims)[-num_words:]
  sim_scores = sim_scores/np.linalg.norm(sim_scores)

  return similar_words, sim_scores

if __name__ == "__main__":
  word = sys.argv[1]
  words, sim_scores = find_close_words(word, [], num_words=20)
  print words
  print sim_scores
