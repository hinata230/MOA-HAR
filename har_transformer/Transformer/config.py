wisdm = {
              "dataset":{
                  "train":8785,
                  "test":2197,
              },
              "model":{
                  "learn_pos_emb":True,
                  "tied_weights":False,
                  "embedding_dim":3, 
                  "transformer_dim":128, 
                  "transformer_hidden_dim":128, 
                  "head_dim":32, 
                  "num_head":2, 
                  "num_layers":2,
                  "vocab_size":256,
                  "max_seq_len":100,
                  "dropout_prob":0.1,
                  "attention_dropout":0.1,
                  "pooling_mode":"MEAN",
                  "num_classes": 6,
              },
              "training":{
                  "batch_size":64, 
                  "learning_rate":0.001,
                  "warmup":800,
                  "lr_decay":"linear",
                  "weight_decay":0,
                  "eval_frequency":50000,
                  "num_train_steps":5000, 
                  "num_init_steps":3000,
                  "num_eval_steps":300, 
                  "patience":10, 
              },
              "extra_attn_config":{
                  "softmax":{"bz_rate":1,},
                  "softmaxRBF32":{"bz_rate":2},
                  "kernelized":{"bz_rate":2},

                  "sketchedRBF32128":{"bz_rate":1, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
                  "skyformer":{"bz_rate":1, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
                  
              }
          }

uci = {
              "dataset":{
                  "train":7352,
                  "test":2947,
              },
              "model":{
                  "learn_pos_emb":True,
                  "tied_weights":False,
                  "embedding_dim":9,
                  "transformer_dim":128,
                  "transformer_hidden_dim":128,
                  "head_dim":32,
                  "num_head":2,
                  "num_layers":2,
                  "vocab_size":256,
                  "max_seq_len":128,
                  "dropout_prob":0.1,
                  "attention_dropout":0.1,
                  "pooling_mode":"MEAN",
                  "num_classes": 6,
              },
              "training":{
                  "batch_size":16,
                  "learning_rate":0.001,
                  "warmup":800,
                  "lr_decay":"linear",
                  "weight_decay":0,
                  "eval_frequency":50000,
                  "num_train_steps":5000,
                  "num_init_steps":3000,
                  "num_eval_steps":300,
                  "patience":10,
              },
              "extra_attn_config":{
                  "softmax":{"bz_rate":1,},
                  "softmaxRBF32":{"bz_rate":2},
                  "kernelized":{"bz_rate":2},

                  "sketchedRBF32128":{"bz_rate":1, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
                  "skyformer":{"bz_rate":1, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},

              }
          }


motionsense = {
              "dataset":{
                  "train":2865,
                  "test":717,
              },
              "model":{
                  "learn_pos_emb":True,
                  "tied_weights":False,
                  "embedding_dim":12,
                  "transformer_dim":256,
                  "transformer_hidden_dim":256,
                  "head_dim":32,
                  "num_head":2,
                  "num_layers":2,
                  "vocab_size":256,
                  "max_seq_len":50,
                  "dropout_prob":0.1,
                  "attention_dropout":0.1,
                  "pooling_mode":"MEAN",
                  "num_classes": 6,
              },
              "training":{
                  "batch_size":16,
                  "learning_rate":0.0005,
                  "warmup":800,
                  "lr_decay":"linear",
                  "weight_decay":0,
                  "eval_frequency":50000,
                  "num_train_steps":4000,
                  "num_init_steps":3000,
                  "num_eval_steps":300,
                  "patience":10,
              },
              "extra_attn_config":{
                  "softmax":{"bz_rate":1,},
                  "softmaxRBF32":{"bz_rate":2},
                  "kernelized":{"bz_rate":2},

                  "sketchedRBF32128":{"bz_rate":1, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
                  "skyformer":{"bz_rate":1, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
              }
          }
moa = {
              "dataset":{
                  "train":16808,
                  "test":4203,
              },
              "model":{
                  "learn_pos_emb":True,
                  "tied_weights":False,
                  "embedding_dim":6,
                  "transformer_dim":256,
                  "transformer_hidden_dim":256,
                  "head_dim":32,
                  "num_head":2,
                  "num_layers":4,
                  "vocab_size":256,
                  "max_seq_len":100,
                  "dropout_prob":0.3,
                  "attention_dropout":0.1,
                  "pooling_mode":"MEAN",
                  "num_classes": 28,
              },
              "training":{
                  "batch_size":256,
                  "learning_rate":0.001,
                  "warmup":20,
                  "lr_decay":"linear",
                  "weight_decay":0.001,
                  "eval_frequency":50000,
                  "num_train_steps":5000,
                  "num_init_steps":3000,
                  "num_eval_steps":10,
                  "patience":10,
              },
              "extra_attn_config":{
                  "softmax":{"bz_rate":1,},
                  "softmaxRBF32":{"bz_rate":2},
                  "kernelized":{"bz_rate":2},

                  "sketchedRBF32128":{"bz_rate":1, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
                  "skyformer":{"bz_rate":1, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},

              }
          }

Config = {
   "UCI":uci,
    "WISDM":wisdm,
    "MOTIONSENSE":motionsense,
    "MOA":moa
}


