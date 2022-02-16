## To Do

- Multichannel Model
    > Model with Multi Channel Input instead of multiple Inputs
- Analysis Framework
    * [ ] Check GradCam
    * [ ] Check Lime
- Check output Format of Spectograms

## Doing

- Dataaugmentation Processor Modules
    > Write Dataaugmentation Modules
    * [x] Pitch Shift
    * [ ] Gain Changer
    * [x] Time Stretch
- Metadata preperation
    * [ ] Shuffel  data with Seed
    * [ ] No Artist appear in multiple data splits
    * [x] Adjust Metadata save

## Done

- Preprocessing Modules create augmented data and/or overwrite base data
- Use Dictionary in DTO
    > Change Datafield for different Splits from List to Dictionary. Adjust Modules accordingly.
    * [x] Spleeter Module
    * [x] STFT Module
    * [x] Datasplit Module
    * [x] Augmentation Modules
- Change to librosa file load because spleeter is doing weird shit
    * [x] Load files with librosa
    * [x] Check Numpy format for Spleeter
    * [x] Check Numpy format for STFT
- Change Split Module
    * [x] Calc samples for desired split length
    * [x] split audio based on possible samples
    * [x] drop rest
- create_class_instance() Bug
- Check Spleeter return value Shape
