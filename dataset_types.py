# Datasets with performance problems
all_dataset_name_list = [
    "ProximalPhalanxOutlineCorrect",
    "LargeKitchenAppliances",
    "CricketY",
    "FaceAll",
    "CricketZ",
    "BeetleFly",
    "CricketX",
    "ChlorineConcentration",
    "Wine",
    "ProximalPhalanxOutlineAgeGroup",
    "ArrowHead",
    "FordB",
    "Adiac",
    "OliveOil",
    "ShapeletSim",
    "CinCECGTorso",
    "PhalangesOutlinesCorrect",
    "Beef",
    "MiddlePhalanxOutlineCorrect",
    "UWaveGestureLibraryX",
    "FiftyWords",
    "Lightning2",
    "Lightning7",
    "Worms",
    "ProximalPhalanxTW",
    "MedicalImages",
    "DistalPhalanxOutlineCorrect",
    "UWaveGestureLibraryZ",
    "DistalPhalanxOutlineAgeGroup",
    "UWaveGestureLibraryY",
    "WordSynonyms",
    "Computers",
    "ElectricDevices",
    "SmallKitchenAppliances",
    "Earthquakes",
    "Ham",
    "WormsTwoClass",
    "DistalPhalanxTW",
    "Herring"
]

short_dataset_name_list = ["Wine", "BeetleFly", "ArrowHead", "OliveOil", "Beef", "Lightning7", "Lightning2",
                     "DistalPhalanxOutlineAgeGroup", "DistalPhalanxTW", "Herring",
                     "ProximalPhalanxTW", "MiddlePhalanxOutlineCorrect", "ProximalPhalanxOutlineCorrect",
                     "ProximalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect", "Ham"]

long_dataset_name_list = [dataset_name for dataset_name in all_dataset_name_list if dataset_name not in short_dataset_name_list]

extra_up_to_2Mb_list = [
    "Chinatown", "SmoothSubspace", "DodgerLoopGame", "DodgerLoopWeekend",
    "DodgerLoopDay", "PickupGestureWiimoteZ", "ShakeGestureWiimoteZ", "ECG200",
    "BirdChicken", "ItalyPowerDemand", "MelbournePedestrian", "FaceFour",
    "SonyAIBORobotSurface1", "MiddlePhalanxOutlineAgeGroup", "MiddlePhalanxTW",
    "Meat", "ToeSegmentation2", "GesturePebbleZ2", "GesturePebbleZ1", "GestureMidAirD2",
    "SonyAIBORobotSurface2", "GestureMidAirD1", "GestureMidAirD3", "Car", "ToeSegmentation1",
    "ShapeletSim", "DiatomSizeReduction", "MoteStrain", "HouseTwenty", "InsectEPGSmallTrain",
    "MedicalImages", "Rock", "Adiac", "InsectEPGRegularTrain", "SwedishLeaf", "Fish",
]

all_up_to_2Mb_dataset_list = [
    "Wine", "BeetleFly", "ArrowHead", "OliveOil", "Beef", "Lightning7", "Lightning2",
    "DistalPhalanxOutlineAgeGroup", "DistalPhalanxTW", "Herring",
    "ProximalPhalanxTW", "MiddlePhalanxOutlineCorrect", "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect", "Ham",
    "Chinatown", "SmoothSubspace", "DodgerLoopGame", "DodgerLoopWeekend",
    "DodgerLoopDay", "PickupGestureWiimoteZ", "ShakeGestureWiimoteZ", "ECG200",
    "BirdChicken", "ItalyPowerDemand", "MelbournePedestrian", "FaceFour",
    "SonyAIBORobotSurface1", "MiddlePhalanxOutlineAgeGroup", "MiddlePhalanxTW",
    "Meat", "ToeSegmentation2", "GesturePebbleZ2", "GesturePebbleZ1", "GestureMidAirD2",
    "SonyAIBORobotSurface2", "GestureMidAirD1", "GestureMidAirD3", "Car", "ToeSegmentation1",
    "ShapeletSim", "DiatomSizeReduction", "MoteStrain", "HouseTwenty", "InsectEPGSmallTrain",
    "MedicalImages", "Rock", "Adiac", "InsectEPGRegularTrain", "SwedishLeaf", "Fish"
]

all_up_to_2Mb_dataset_list_remaining = [
    "ProximalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect", "Ham",
    "Chinatown", "SmoothSubspace", "DodgerLoopGame", "DodgerLoopWeekend",
    "DodgerLoopDay", "PickupGestureWiimoteZ", "ShakeGestureWiimoteZ", "ECG200",
    "BirdChicken", "ItalyPowerDemand", "MelbournePedestrian", "FaceFour",
    "SonyAIBORobotSurface1", "MiddlePhalanxOutlineAgeGroup", "MiddlePhalanxTW",
    "Meat", "ToeSegmentation2", "GesturePebbleZ2", "GesturePebbleZ1", "GestureMidAirD2",
    "SonyAIBORobotSurface2", "GestureMidAirD1", "GestureMidAirD3", "Car", "ToeSegmentation1",
    "ShapeletSim", "DiatomSizeReduction", "MoteStrain", "HouseTwenty", "InsectEPGSmallTrain",
    "MedicalImages", "Rock", "Adiac", "InsectEPGRegularTrain", "SwedishLeaf", "Fish"
]