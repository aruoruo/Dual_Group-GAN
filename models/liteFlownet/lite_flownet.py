#!/usr/bin/env python
import torch
import math
import torch.nn
from models.liteFlownet import correlation
import torch.nn.functional as F

Backward_Grid = {}


def Backward(tensorInput, tensorFlow):
    if str(tensorFlow.size()) not in Backward_Grid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(
            tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(
            tensorFlow.size(0), -1, -1, tensorFlow.size(3))

        Backward_Grid[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], 1)
    # end

    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                            tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

    return F.grid_sample(input=tensorInput,
                         grid=(Backward_Grid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1),
                         mode='bilinear', padding_mode='zeros')


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        class Features(torch.nn.Module):
            def __init__(self):
                super(Features, self).__init__()

                self.moduleOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

            # end

            def forward(self, tensorInput):
                tensorOne = self.moduleOne(tensorInput)
                tensorTwo = self.moduleTwo(tensorOne)
                tensorThr = self.moduleThr(tensorTwo)
                tensorFou = self.moduleFou(tensorThr)
                tensorFiv = self.moduleFiv(tensorFou)
                tensorSix = self.moduleSix(tensorFiv)

                return [tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix]

        class Matching(torch.nn.Module):
            def __init__(self, intLevel):
                super(Matching, self).__init__()

                self.dblBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]

                if intLevel != 2:
                    self.moduleFeat = torch.nn.Sequential()

                elif intLevel == 2:
                    self.moduleFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )

                # end

                if intLevel == 6:
                    self.moduleUpflow = None

                elif intLevel != 6:
                    self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2,
                                                                 padding=1, bias=False, groups=2)

                # end

                if intLevel >= 4:
                    self.moduleUpcorr = None

                elif intLevel < 4:
                    self.moduleUpcorr = torch.nn.ConvTranspose2d(in_channels=49, out_channels=49, kernel_size=4,
                                                                 stride=2, padding=1, bias=False, groups=49)

                # end

                self.moduleMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=49, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel],
                                    stride=1, padding=[0, 0, 3, 2, 2, 1, 1][intLevel])
                )

            # end

            def forward(self, tensorFirst, tensorSecond, tensorFeaturesFirst, tensorFeaturesSecond, tensorFlow):
                tensorFeaturesFirst = self.moduleFeat(tensorFeaturesFirst)
                tensorFeaturesSecond = self.moduleFeat(tensorFeaturesSecond)

                if tensorFlow is not None:
                    tensorFlow = self.moduleUpflow(tensorFlow)
                # end

                if tensorFlow is not None:
                    tensorFeaturesSecond = Backward(tensorInput=tensorFeaturesSecond,
                                                    tensorFlow=tensorFlow * self.dblBackward)
                # end

                if self.moduleUpcorr is None:
                    tensorCorrelation = F.leaky_relu(
                        input=correlation.FunctionCorrelation(tensorFirst=tensorFeaturesFirst,
                                                              tensorSecond=tensorFeaturesSecond, intStride=1),
                        negative_slope=0.1, inplace=False)

                elif self.moduleUpcorr is not None:
                    tensorCorrelation = self.moduleUpcorr(F.leaky_relu(
                        input=correlation.FunctionCorrelation(tensorFirst=tensorFeaturesFirst,
                                                              tensorSecond=tensorFeaturesSecond, intStride=2),
                        negative_slope=0.1, inplace=False))

                # end

                return (tensorFlow if tensorFlow is not None else 0.0) + self.moduleMain(tensorCorrelation)

        # end

        # end

        class Subpixel(torch.nn.Module):
            def __init__(self, intLevel):
                super(Subpixel, self).__init__()

                self.dblBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]

                if intLevel != 2:
                    self.moduleFeat = torch.nn.Sequential()

                elif intLevel == 2:
                    self.moduleFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )

                # end

                self.moduleMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=[0, 0, 130, 130, 194, 258, 386][intLevel], out_channels=128,
                                    kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel],
                                    stride=1, padding=[0, 0, 3, 2, 2, 1, 1][intLevel])
                )

            # end

            def forward(self, tensorFirst, tensorSecond, tensorFeaturesFirst, tensorFeaturesSecond, tensorFlow):
                tensorFeaturesFirst = self.moduleFeat(tensorFeaturesFirst)
                tensorFeaturesSecond = self.moduleFeat(tensorFeaturesSecond)

                if tensorFlow is not None:
                    tensorFeaturesSecond = Backward(tensorInput=tensorFeaturesSecond,
                                                    tensorFlow=tensorFlow * self.dblBackward)
                # end

                return (tensorFlow if tensorFlow is not None else 0.0) + self.moduleMain(
                    torch.cat([tensorFeaturesFirst, tensorFeaturesSecond, tensorFlow], 1))

        # end

        # end

        class Regularization(torch.nn.Module):
            def __init__(self, intLevel):
                super(Regularization, self).__init__()

                self.dblBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]

                self.intUnfold = [0, 0, 7, 5, 5, 3, 3][intLevel]

                if intLevel >= 5:
                    self.moduleFeat = torch.nn.Sequential()

                elif intLevel < 5:
                    self.moduleFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=[0, 0, 32, 64, 96, 128, 192][intLevel], out_channels=128,
                                        kernel_size=1, stride=1, padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )

                # end

                self.moduleMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=[0, 0, 131, 131, 131, 131, 195][intLevel], out_channels=128,
                                    kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                if intLevel >= 5:
                    self.moduleDist = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                                        kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel], stride=1,
                                        padding=[0, 0, 3, 2, 2, 1, 1][intLevel])
                    )

                elif intLevel < 5:
                    self.moduleDist = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                                        kernel_size=([0, 0, 7, 5, 5, 3, 3][intLevel], 1), stride=1,
                                        padding=([0, 0, 3, 2, 2, 1, 1][intLevel], 0)),
                        torch.nn.Conv2d(in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                                        out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                                        kernel_size=(1, [0, 0, 7, 5, 5, 3, 3][intLevel]), stride=1,
                                        padding=(0, [0, 0, 3, 2, 2, 1, 1][intLevel]))
                    )

                # end

                self.moduleScaleX = torch.nn.Conv2d(in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], out_channels=1,
                                                    kernel_size=1, stride=1, padding=0)
                self.moduleScaleY = torch.nn.Conv2d(in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], out_channels=1,
                                                    kernel_size=1, stride=1, padding=0)

            # eny

            def forward(self, tensorFirst, tensorSecond, tensorFeaturesFirst, tensorFeaturesSecond, tensorFlow):
                tensorDifference = (tensorFirst - Backward(tensorInput=tensorSecond,
                                                           tensorFlow=tensorFlow * self.dblBackward)).pow(2.0).sum(1,
                                                                                                                   True).sqrt()

                tensorDist = self.moduleDist(self.moduleMain(torch.cat([tensorDifference,
                                                                        tensorFlow - tensorFlow.view(tensorFlow.size(0),
                                                                                                     2, -1).mean(2,
                                                                                                                 True).view(
                                                                            tensorFlow.size(0), 2, 1, 1),
                                                                        self.moduleFeat(tensorFeaturesFirst)], 1)))
                tensorDist = tensorDist.pow(2.0).neg()
                tensorDist = (tensorDist - tensorDist.max(1, True)[0]).exp()

                tensorDivisor = tensorDist.sum(1, True).reciprocal()

                tensorScaleX = self.moduleScaleX(
                    tensorDist * F.unfold(input=tensorFlow[:, 0:1, :, :], kernel_size=self.intUnfold,
                                          stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(
                        tensorDist)) * tensorDivisor
                tensorScaleY = self.moduleScaleY(
                    tensorDist * F.unfold(input=tensorFlow[:, 1:2, :, :], kernel_size=self.intUnfold,
                                          stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(
                        tensorDist)) * tensorDivisor

                return torch.cat([tensorScaleX, tensorScaleY], 1)

        # end

        # end

        self.moduleFeatures = Features()
        self.moduleMatching = torch.nn.ModuleList([Matching(intLevel) for intLevel in [2, 3, 4, 5, 6]])
        self.moduleSubpixel = torch.nn.ModuleList([Subpixel(intLevel) for intLevel in [2, 3, 4, 5, 6]])
        self.moduleRegularization = torch.nn.ModuleList([Regularization(intLevel) for intLevel in [2, 3, 4, 5, 6]])

    # self.load_state_dict(torch.load('./network-' + arguments_strModel + '.pytorch'))
    # end

    def forward(self, tensorFirst, tensorSecond):
        tensorFirst[:, 0, :, :] = tensorFirst[:, 0, :, :] - 0.411618
        tensorFirst[:, 1, :, :] = tensorFirst[:, 1, :, :] - 0.434631
        tensorFirst[:, 2, :, :] = tensorFirst[:, 2, :, :] - 0.454253

        tensorSecond[:, 0, :, :] = tensorSecond[:, 0, :, :] - 0.410782
        tensorSecond[:, 1, :, :] = tensorSecond[:, 1, :, :] - 0.433645
        tensorSecond[:, 2, :, :] = tensorSecond[:, 2, :, :] - 0.452793

        tensorFeaturesFirst = self.moduleFeatures(tensorFirst)
        tensorFeaturesSecond = self.moduleFeatures(tensorSecond)

        tensorFirst = [tensorFirst]
        tensorSecond = [tensorSecond]

        for intLevel in [1, 2, 3, 4, 5]:
            tensorFirst.append(F.interpolate(input=tensorFirst[-1], size=(
                tensorFeaturesFirst[intLevel].size(2), tensorFeaturesFirst[intLevel].size(3)), mode='bilinear',
                                             align_corners=False))
            tensorSecond.append(F.interpolate(input=tensorSecond[-1], size=(
                tensorFeaturesSecond[intLevel].size(2), tensorFeaturesSecond[intLevel].size(3)), mode='bilinear',
                                              align_corners=False))
        # end

        tensorFlow = None

        for intLevel in [-1, -2, -3, -4, -5]:
            tensorFlow = self.moduleMatching[intLevel](tensorFirst[intLevel], tensorSecond[intLevel],
                                                       tensorFeaturesFirst[intLevel], tensorFeaturesSecond[intLevel],
                                                       tensorFlow)
            tensorFlow = self.moduleSubpixel[intLevel](tensorFirst[intLevel], tensorSecond[intLevel],
                                                       tensorFeaturesFirst[intLevel], tensorFeaturesSecond[intLevel],
                                                       tensorFlow)
            tensorFlow = self.moduleRegularization[intLevel](tensorFirst[intLevel], tensorSecond[intLevel],
                                                             tensorFeaturesFirst[intLevel],
                                                             tensorFeaturesSecond[intLevel], tensorFlow)
        # end

        return tensorFlow * 20.0

    @staticmethod
    def batch_estimate(tensor_batch, moduleNetwork):
        tensor_batch = (tensor_batch + 1) / 2  # normalize to 0 ~ 1.
        intWidth = tensor_batch.size(3)
        intHeight = tensor_batch.size(2)
        tensorFirst = tensor_batch[:, :3, ...]
        tensorSecond = tensor_batch[:, 3:, ...]

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

        tensorPreprocessedFirst = F.interpolate(input=tensorFirst,
                                                size=(intPreprocessedHeight, intPreprocessedWidth),
                                                mode='bilinear', align_corners=False)
        tensorPreprocessedSecond = F.interpolate(input=tensorSecond,
                                                 size=(intPreprocessedHeight, intPreprocessedWidth),
                                                 mode='bilinear', align_corners=False)

        tensorFlow = F.interpolate(input=moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond),
                                   size=(intHeight, intWidth), mode='bilinear', align_corners=False)

        tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        return tensorFlow
