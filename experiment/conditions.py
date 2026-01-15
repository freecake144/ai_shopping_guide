class ExperimentConditions:
    # 定义 2x2 矩阵
    GROUPS = {
        'A': {'adaptability': 'HIGH', 'calibration': 'HIGH'},
        'B': {'adaptability': 'HIGH', 'calibration': 'LOW'},
        'C': {'adaptability': 'LOW',  'calibration': 'HIGH'},
        'D': {'adaptability': 'LOW',  'calibration': 'LOW'}
    }

    @staticmethod
    def get_config(group_id):
        return ExperimentConditions.GROUPS.get(group_id)