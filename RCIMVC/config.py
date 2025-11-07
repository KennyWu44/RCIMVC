data_dict = {
    0: 'Scene-15',
    3: 'MSRC_v1',
    6: 'MFeat',
    7: 'cub_googlenet', 
    8: 'CCV',
    9: 'ORL',
    11: 'CCV-3',
    14: 'ORL-4',
    15: 'Scene-15-3',
}


def get_config(flag=1):
    """Determine the parameter information of the network"""
    data_name = data_dict[flag]
    if data_name in ['Scene-15']:
        return dict(
            dataset=data_name,
            topk=10,
            missing_rate=0.0,
            n_clusters=15,
            training=dict(
                lr=1.0e-3,
                epoch=500,
                batch_size=1024,
            ),
            Autoencoder=dict(
                gcnEncoder1=[40, 1024, 1024, 1024, 1024 // 2],
                gcnEncoder2=[59, 1024, 1024, 1024, 1024 // 2],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            )

        )
    elif data_name in ['Scene-15-3']:
        return dict(
            dataset=data_name,
            topk=10,
            missing_rate=0.0,
            n_clusters=15,
            training=dict(
                lr=1.0e-3,
                epoch=500,
                batch_size=1024,
            ),
            Autoencoder=dict(
                gcnEncoder1=[40, 1024, 1024, 1024, 1024 // 2],
                gcnEncoder2=[59, 1024, 1024, 1024, 1024 // 2],
                gcnEncoder3=[20, 1024, 1024, 1024, 1024 // 2],
                activations1='relu',
                activations2='relu',
                activations3='relu',
                batchnorm=True,
            )

        )
    elif data_name in ['cub_googlenet']:
        return dict(
            dataset=data_name,
            topk=10,
            missing_rate=0.0,
            n_clusters=10,
            training=dict(
                lr=1.0e-3,
                epoch=500,
                batch_size=2048,
            ),
            Autoencoder=dict(
                gcnEncoder1=[1024, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder2=[300, 1024, 1024, 1024, 1024 // 8],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
        )
    elif data_name in ['MSRC_v1']:
        return dict(
            dataset=data_name,
            topk=10,
            missing_rate=0.0,
            n_clusters=7,
            training=dict(
                lr=1.0e-3,
                epoch=500,
                batch_size=256,
            ),
            Autoencoder=dict(
                gcnEncoder1=[576, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder2=[512, 1024, 1024, 1024, 1024 // 8],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
        )
    
    elif data_name in ['CCV']:
        return dict(
            dataset=data_name,
            topk=10,
            missing_rate=0.0,
            n_clusters=20,
            training=dict(
                lr=1.0e-3,
                epoch=500,
                batch_size=2048,
            ),
            Autoencoder=dict(
                gcnEncoder1=[5000, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder2=[5000, 1024, 1024, 1024, 1024 // 8],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
        )
    elif data_name in ['CCV-3']:
        return dict(
            dataset=data_name,
            topk=10,
            missing_rate=0.0,
            n_clusters=20,
            training=dict(
                lr=1.0e-3,
                epoch=500,
                batch_size=2048,
            ),
            Autoencoder=dict(
                gcnEncoder1=[5000, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder2=[5000, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder3=[4000, 1024, 1024, 1024, 1024 // 8],
                activations1='relu',
                activations2='relu',
                activations3='relu',
                batchnorm=True,
            ),
        )
 
    elif data_name in ['MFeat']:
        return dict(
            dataset=data_name,
            topk=10,
            missing_rate=0.0,
            n_clusters=10,
            training=dict(
                lr=1.0e-3,
                epoch=500,
                batch_size=2048,
            ),
            Autoencoder=dict(
                gcnEncoder1=[216, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder2=[240, 1024, 1024, 1024, 1024 // 8],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
        )
 
    elif data_name in ['ORL']:
        return dict(
            dataset=data_name,
            topk=10,
            missing_rate=0.0,
            n_clusters=40,
            training=dict(
                lr=1.0e-3,
                epoch=500,
                batch_size=2048,
            ),
            Autoencoder=dict(
                gcnEncoder1=[512, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder2=[864, 1024, 1024, 1024, 1024 // 8],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
        )
    
    elif data_name in ['ORL-4']:
        return dict(
            dataset=data_name,
            topk=10,
            missing_rate=0.0,
            n_clusters=40,
            training=dict(
                lr=1.0e-3,
                epoch=500,
                batch_size=2048,
            ),
            Autoencoder=dict(
                gcnEncoder1=[512, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder2=[864, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder3=[254, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder4=[59, 1024, 1024, 1024, 1024 // 8],

                activations1='relu',
                activations2='relu',
                activations3='relu',
                activations4='relu',

                batchnorm=True,
            ),
        )

    
    