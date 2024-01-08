"""Dataset info file for 3D POP"""



dataset_info = dict(
    dataset_name='3dpop',
    paper_info=dict(
        author= "Alex Chan",
        title='3D POP',
        container='CVPR',
        year='2023'
    ),

    keypoint_info={
        0:
        dict(name='hd_beak', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='hd_nose',
            id=1,
            color=[51, 153, 255],
            type='upper'),
        2:
        dict(
            name='hd_leftEye',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='hd_rightEye'),
        3:
        dict(
            name='hd_rightEye',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='hd_leftEye'),
        4:
        dict(
            name='bp_leftShoulder',
            id=4,
            color=[51, 153, 255],
            type='lower',
            swap='bp_rightShoulder'),
        5:
        dict(
            name='bp_rightShoulder',
            id=5,
            color=[0, 255, 0],
            type='lower',
            swap='bp_leftShoulder'),
        6:
        dict(
            name='bp_topKeel',
            id=6,
            color=[255, 128, 0],
            type='lower'),
        7:
        dict(
            name='bp_bottomKeel',
            id=7,
            color=[0, 255, 0],
            type='lower'),
        8:
        dict(
            name='bp_tail',
            id=8,
            color=[255, 128, 0],
            type='lower')
    },

    skeleton_info={
        0:
        dict(link=('hd_beak', 'hd_nose'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('hd_nose', 'hd_leftEye'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('hd_nose', 'hd_rightEye'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('bp_leftShoulder', 'bp_rightShoulder'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('bp_leftShoulder', 'bp_tail'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('bp_rightShoulder', 'bp_tail'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('bp_leftShoulder', 'bp_topKeel'), id=6, color=[51, 153, 255]),
        7:
        dict(
            link=('bp_rightShoulder', 'bp_topKeel'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(link=('bp_topKeel', 'bp_bottomKeel'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('bp_bottomKeel', 'bp_tail'), id=9, color=[255, 128, 0])
    },

    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.,1.
    ],

    sigmas=[
        0.02, 0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02
    ]
    )