import argparse
import struct
import numpy as np
from sklearn.metrics import pairwise as skmetrics
import sys

FLOAT_SIZE = 4

class w2v:
  def __init__(self):
    self.binFile="../word_similarity/vectors-cbow-bnc+ukwac+wikipedia.bin"
    self.vocab=None
    self.matrix=None

  def readVectors(self):
    # vocab is a list containing the row info
    # matrix will have data
    infd=open(self.binFile,"r")
    header = infd.readline().rstrip()
    vocab_s, dims = map(int, header.split(" "))
    self.vocab = []
    # init matrix
    self.matrix = np.zeros((vocab_s, dims), dtype=np.float)
    i = 0
    while True:
    #while i<7:
        line = infd.readline()
	if not line:
	    break
        sep = line.find(" ")
        word = line[:sep]
        data = line[sep+1:]
        if len(data) < FLOAT_SIZE * dims + 1:
            data += infd.read(FLOAT_SIZE * dims + 1 - len(data))
        data = data[:-1]
        self.vocab.append(word)
        vector = (struct.unpack("%df" % dims, data))
        self.matrix[i] = vector
        i += 1
    infd.close()

  def reduce_vectors(self, word_list, pos):
  #reduce the vectors so that they only include the vectors in the lexical list
    #kind of hacky, but assume a noun unless noun does not exist, then assume adjective, then assume verb, then skip
    vocab = self.vocab
    new_vocab = []
    new_matrix = np.zeros((len(word_list), self.matrix.shape[1]), dtype=np.float)
    for new_idx, word in enumerate(word_list):
      if word + pos in self.vocab:
        old_idx = vocab.index(word+'-n') 
        new_vocab.append(word+'-n')
      else:
        old_idx = None
        new_vocab.append(None)
      if old_idx:
        new_matrix[new_idx, :] = self.matrix[old_idx,:]
      else:
        print "Word %s%s not in word2vec.\n" %(word, pos)
        new_matrix[new_idx,:] = -1000000 #this should make this vector far from everything
    self.matrix = new_matrix
    self.vocab = new_vocab 

  def printSample(self):
    for i in range(0,7):
      print self.vocab[i]+"\t"
      #print str(self.matrix[i])+"\n"

  def getCosDistanceN(self,noun1,noun2):
    if(not self.vocab.__contains__(noun1+'-n')):
      print "Vocab does not contain "+noun1+'-n'
      return 0
    if(not self.vocab.__contains__(noun2+'-n')):
      print "Vocab does not contain "+noun2+'-n'
      return 0
    ind1=self.vocab.index(noun1+'-n')
    ind2=self.vocab.index(noun2+'-n')
    dist=skmetrics.cosine_similarity(self.matrix[ind1],self.matrix[ind2])
    return dist[0][0]

  def getCosDistanceV(self,verb1,verb2):
    if(not self.vocab.__contains__(verb1+'-v')):
      print "Vocab does not contain "+verb1+'-v'
      return 0
    if(not self.vocab.__contains__(verb2+'-v')):
      print "Vocab does not contain "+verb2+'-v'
      return 0
    ind1=self.vocab.index(verb1+'-v')
    ind2=self.vocab.index(verb2+'-v')
    dist=skmetrics.cosine_similarity(self.matrix[ind1],self.matrix[ind2])
    return dist[0][0]
  
  def findClosestWords(self, word, pos='-n'):
    if(not self.vocab.__contains__(word + pos)):
      print "Vocab does not contain "+ word + pos
      return 0
    if(not self.vocab.__contains__(word + pos)):
      print "Vocab does not contain "+ word + pos
      return 0
    ind1=self.vocab.index(word + pos)
    numerator = np.dot(self.matrix, self.matrix[ind1])
    denominator = np.linalg.norm(self.matrix[ind1])*np.linalg.norm(self.matrix, axis=1)
    dist = numerator/denominator
    return dist
#    vocab_idx = np.argsort(dist)[-10:]
#    return [self.vocab[idx] for idx in vocab_idx], dist[vocab_idx]

def demo1():
  parser = argparse.ArgumentParser(
              description="Converts a Mikolov binary vector file into one compatible with Trento's COMPOSES.")
  parser.add_argument('--input', '-i', help='Input file')
  parser.add_argument('--output', '-o', type=argparse.FileType('w'), help='Output file')
  args = parser.parse_args()
  W2V=w2v()
  W2V.readVectors()
  W2V.printSample()
  w1='time'
  w2='year'
  dist=W2V.getCosDistanceN(w1,w2)
  print "cos sim b/w "+w1+" and "+w2+" = "+str(dist)

def main():
  if len(sys.argv) > 1:
    word = sys.argv[1]
  else:
    word = 'zebra'
  W2V=w2v()
  W2V.readVectors()
  similar_words, similarity=W2V.findClosestWords(word)
  similar_words.reverse()
  for iw, w in enumerate(similar_words):
    print "Closest words are: %s:%f.\n" %(w, similarity[-(iw+1)])
    
if __name__=="__main__":
	main()
