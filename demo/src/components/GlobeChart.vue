<template lang="pug">
  div.outer-container
    div.chart-container(ref="globe")
    div.tool-bar
      el-select(v-model="typeSelected" placeholder="Capacity Type" @change="handleChangeOption" :disabled="loading||playing")
        el-option(v-for="item in typeOptions" :key="item.value" :label="item.label" :value="item.value" :disabled="loading||playing")
      el-cascader(v-model="paramSelected" :options="selectOptions" placeholder="Select Param T" @change="handleChangeParam" :disabled="loading||playing")
      el-date-picker(v-model="dateSelected" :pickerOptions="pickerOptions" :clearable="false" @change="handleChangeDate" :disabled="loading||playing")
      el-button(:icon="playing?'el-icon-switch-button':'el-icon-caret-right'" circle @click="handlePlayDate" :disabled="dateSelected.getTime() >= endDate.getTime()")
</template>

<script>
import echarts from 'echarts'
import 'echarts-gl'
import axios from 'axios'

async function getCalcData(T, ep, date) {
  const paramStr = `T=${T} ep=${ep}`;
  const fileStr = `${date.getFullYear()}-${date.getMonth() + 1}-${date.getDate()}.json`
  const capacityMatrixDir = `/data/capacity/${paramStr}/${fileStr}`
  const severityVectorDir = `/data/severity/${fileStr}`

  const capacityMatrix = (await axios.get(capacityMatrixDir)).data;
  const severityVector = (await axios.get(severityVectorDir)).data;

  return {capacityMatrix, severityVector};
}

function timeout(time) {
  return new Promise(resolve => {
    setTimeout(function () {
      resolve()
    }, time)
  })
}

function getNextDate(date) {
  const oneDayDelta = 1000 * 3600 * 24;
  return new Date(date.getTime() + oneDayDelta);
}

function getFlight(route, lineWidth, direction, showEffect = true) {
  return {
    type: 'lines3D',
    effect: {
      show: showEffect,
      constantSpeed: lineWidth / 1.5,
      trailWidth: Math.max(0.3 * lineWidth, 1),
      trailLength: 0.1,
      trailOpacity: 1,
      trailColor: direction ? 'rgb(113, 150, 60)' : 'rgb(188, 59, 24)'
    },

    lineStyle: {
      width: lineWidth,
      color: 'rgb(50, 50, 150)',
      opacity: 0.1
    },
    blendMode: 'lighter',

    data: [route]
  }
}

function getPoint(airport, pointSize) {
  return {
    type: 'scatter3D',
    coordinateSystem: 'globe',
    blendMode: 'lighter',
    symbolSize: pointSize,
    itemStyle: {
      color: 'rgb(150, 50, 150)',
      opacity: 0.9
    },
    data: [airport]
  }
}

export default {
  name: "GlobeChart",
  props: {
    startDate: {
      type: Date,
      default: () => new Date(2020, 1, 16)
    },
    endDate: {
      type: Date,
      default: () => new Date(2020, 3, 16)
    }
  },
  data() {
    return {
      waitList: Promise.resolve(),
      intervalIdx: null,
      playInterval: 150,
      expectedMaxS: 50000,
      toggled: false,
      loading: false,
      playing: false,
      paramSelected: [1000, 0.7],
      dateSelected: this.startDate,
      typeSelected: 'relative',
      pickerOptions: {
        disabledDate: (time) => {
          return time.getTime() < this.startDate.getTime() || time.getTime() > this.endDate.getTime()
        },
        shortcuts: [{
          text: 'Feb. 16, 2020',
          onClick: (picker) => {
            picker.$emit('pick', this.startDate);
          }
        }, {
          text: 'Apr. 16, 2020',
          onClick: (picker) => {
            picker.$emit('pick', this.endDate);
          }
        }]
      },
      typeOptions: [
        {
          value: 'relative',
          label: 'Relative Capacity'
        },
        {
          value: 'absolute',
          label: 'Absolute Capacity'
        }
      ],
      selectOptions: [{
        value: 100,
        label: 'T=100',
        disabled: true
      }, {
        value: 1000,
        label: 'T=1000',
        children: [{
          value: 0.7,
          label: 'ep=0.7'
        }, {
          value: 0.8,
          label: 'ep=0.8'
        }, {
          value: 0.9,
          label: 'ep=0.9'
        }]
      }, {
        value: 10000,
        label: 'T=10000',
        disabled: true
      }],
      pointScale: 30,
      lineScale: 15,
      chart: null,
      airports: null,
      maxCapacityMatrix: null,
      option: {
        animation: false,
        globe: {
          baseTexture: "/data/geoinfo/texture/world.topo.bathy.200401.jpg",
          heightTexture: "/data/geoinfo/texture/world.topo.bathy.200401.jpg",
          displacementScale: 0.04,
          displacementQuality: 'medium',
          shading: 'lambert',
          environment: "/data/geoinfo/texture/starfield.jpg",
          realisticMaterial: {
            roughness: 0.9
          },
          postEffect: {
            enable: true,
            depthOfField: {
              enable: false,
              focalDistance: 150
            }
          },
          temporalSuperSampling: {
            enable: true
          },
          light: {
            ambient: {
              intensity: 0
            },
            main: {
              intensity: 5,
              shadow: true,
              time: (() => {
                const date = new Date();
                date.setHours(20);
                return date
              })()
            },
            ambientCubemap: {
              texture: "/data/geoinfo/texture/pisa.hdr",
              exposure: 1,
              diffuseIntensity: 0.2,
              specularIntensity: 2
            }
          },
          viewControl: {
            targetCoord: [112.5, 31],
            autoRotate: false,
            distance: 50
          }
        },
        series: null
      }
    }
  },
  methods: {
    chartNextTick() {
      return new Promise(resolve => {
        this.chart.on('rendered', () => {
          resolve();
        });
      });
    },

    async updateChart(showEffect = true) {
      this.loading = true;
      // this.chart.showLoading();
      await this.updateSeries(this.paramSelected[0], this.paramSelected[1], this.dateSelected, this.typeSelected, showEffect);
      this.chart.setOption(this.option, true);
      await this.chartNextTick();
      this.loading = false;
      this.toggled = false;
      // this.chart.hideLoading();
    },

    async handlePlayDate() {
      if (this.playing) {
        this.playing = false;
        // set this.playing as false first
        await this.chartNextTick();
        await this.updateChart(true);
      } else {
        if (this.dateSelected.getTime() < this.endDate.getTime()) {
          this.playing = true;
          while (this.playing && this.dateSelected.getTime() < this.endDate.getTime()) {
            this.dateSelected = getNextDate(this.dateSelected);
            if (!this.playing) {
              return;
            }
            if (this.dateSelected.getTime() < this.endDate.getTime()) {
              await this.updateChart(false);
            } else {
              await this.updateChart(true);
              this.playing = false;
            }
            await timeout(this.playInterval);
          }
        }
      }
    },

    handleChangeOption() {
      this.updateChart();
    },

    handleChangeParam() {
      this.updateChart();
    },

    handleChangeDate() {
      this.updateChart();
    },

    initChart() {
      const globe = this.$refs['globe'];
      this.chart = echarts.init(globe);
      this.chart.setOption(this.option);
    },

    toggle() {
      this.toggled = !this.toggled;
      this.option.series && this.option.series.forEach((_, idx) => {
        this.chart.dispatchAction({
          type: 'lines3DToggleEffect',
          seriesIndex: idx
        });
      })
    },

    stopOrStartAnime(event) {
      if (event.key === ' ' && !this.playing) {
        this.toggle();
      }
    },

    async updateSeries(T, ep, date, type = 'relative', showEffect = true) {

      this.option.series = [];
      const {capacityMatrix, severityVector} = await getCalcData(T, ep, date);
      const range = this.airports.length

      for (let i = 0; i < range; i++) {
        for (let j = 0; j < range; j++) {
          if (i !== j && capacityMatrix[i][j] !== 0) {
            const route = [this.airports[i], this.airports[j]];
            let lineWidth = 1;
            if (type === 'relative') {
              lineWidth = Math.max(this.lineScale * capacityMatrix[i][j] / this.maxCapacityMatrix[i][j], 1);
            } else if (type === 'absolute') {
              const maxC = Math.max.apply(null, capacityMatrix.flat(2));
              lineWidth = Math.max(this.lineScale * capacityMatrix[i][j] / maxC, 1);
            }
            const flight = getFlight(route, lineWidth, i < j, showEffect);
            this.option.series.push(flight);
          }
        }
      }

      //Add the lines first
      for (let i = 0; i < range; i++) {
        const maxS = Math.max.apply(null, severityVector);
        const scaleRate = 1 / (1 + Math.log10(this.expectedMaxS / maxS))
        const pointSize = Math.max(scaleRate * this.pointScale / (1 + Math.log10(maxS / severityVector[i])), 1);
        const point = getPoint(this.airports[i], pointSize);
        this.option.series.push(point);
      }
    }
  },

  async beforeMount() {
    this.airports = (await axios.get('/data/geoinfo/flights/airports.json')).data;
    this.maxCapacityMatrix = (await axios.get('/data/capacity/max.json')).data;
  },

  async mounted() {
    await this.$nextTick();
    this.initChart();
    await this.updateChart();
    window.addEventListener('resize', this.chart.resize);
    window.addEventListener('keydown', this.stopOrStartAnime);
  },

  beforeDestroy() {
    window.removeEventListener('resize', this.chart.resize);
    window.removeEventListener('keydown', this.stopOrStartAnime);
  }
}
</script>

<style lang="less" scoped>
.outer-container {
  width: 60%;
  min-width: 900px;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.tool-bar {
  margin-top: 1rem;

  * {
    margin-right: 1.5rem
  }
}

.chart-container {
  width: 65%;
  min-width: 830px;
  height: 40vh;
  min-height: 600px;
  //background-color: red
}
</style>