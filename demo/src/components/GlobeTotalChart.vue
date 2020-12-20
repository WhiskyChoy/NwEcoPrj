<template lang="pug">
  div.outer-container
    div.chart-container(ref="globe")
</template>

<script>
import axios from "axios";
import echarts from "echarts";

let option = {
  legend: {
    selectedMode: 'single',
    left: 'left',
    data: null,
    orient: 'vertical',
    textStyle: {
      color: '#fff'
    }
  },
  globe: {

    environment: '/data/geoinfo/texture/starfield.jpg',
    baseTexture: "/data/geoinfo/texture/world.topo.bathy.200401.jpg",
    heightTexture: '/data/geoinfo/texture/world.topo.bathy.200401.jpg',

    displacementScale: 0.04,
    displacementQuality: 'medium',

    shading: 'lambert',
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
        texture: '/data/geoinfo/texture/pisa.hdr',
        exposure: 1,
        diffuseIntensity: 0.2,
        specularIntensity: 2
      }
    },
    viewControl: {
      autoRotate: true
    },
    silent: true
  },
  series: null
}

export default {
  name: "GlobeTotalChart",
  data() {
    return {
      toggled: false,
      chart: null,
      option
    }
  },
  methods:{
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
  },

  async mounted() {
    const data = (await axios.get('/data/geoinfo/flights/total-flights.json')).data;
    let airports = data.airports.map(function (item) {
      return {
        coord: [item[3], item[4]]
      }
    });

    function getAirportCoord(idx) {
      return [data.airports[idx][3], data.airports[idx][4]];
    }

    // Route: [airlineIndex, sourceAirportIndex, destinationAirportIndex]
    let routesGroupByAirline = {};
    data.routes.forEach(function (route) {
      let airline = data.airlines[route[0]];
      let airlineName = airline[0];
      if (!routesGroupByAirline[airlineName]) {
        routesGroupByAirline[airlineName] = [];
      }
      routesGroupByAirline[airlineName].push(route);
    });

    let pointsData = [];
    data.routes.forEach(function (airline) {
      pointsData.push(getAirportCoord(airline[1]));
      pointsData.push(getAirportCoord(airline[2]));
    });

    let series = data.airlines.map(function (airline) {
      let airlineName = airline[0];
      let routes = routesGroupByAirline[airlineName];

      if (!routes) {
        return null;
      }
      return {
        type: 'lines3D',
        name: airlineName,

        effect: {
          show: true,
          trailWidth: 2,
          trailLength: 0.15,
          trailOpacity: 1,
          trailColor: 'rgb(30, 30, 60)'
        },

        lineStyle: {
          width: 1,
          color: 'rgb(50, 50, 150)',
          // color: 'rgb(118, 233, 241)',
          opacity: 0.1
        },
        blendMode: 'lighter',

        data: routes.map(function (item) {
          return [airports[item[1]].coord, airports[item[2]].coord];
        })
      };
    }).filter(function (series) {
      return !!series;
    });
    series.push({
      type: 'scatter3D',
      coordinateSystem: 'globe',
      blendMode: 'lighter',
      symbolSize: 10 * Math.random(),
      itemStyle: {
        color: 'rgb(50, 50, 150)',
        opacity: 0.2
      },
      data: pointsData
    });
    this.option.series = series;
    this.option.legend.data = Object.keys(routesGroupByAirline);
    const globe = this.$refs['globe'];
    this.chart = echarts.init(globe);
    this.chart.setOption(this.option);
    window.addEventListener('resize', this.chart.resize);
    window.addEventListener('keydown', this.stopOrStartAnime);
  },
  beforeDestroy() {
    window.removeEventListener('resize', this.chart.resize);
    window.removeEventListener('keydown', this.stopOrStartAnime);
  }

}
</script>

<style scoped>
.outer-container {
  width: 60%;
  min-width: 900px;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.chart-container {
  width: 65%;
  min-width: 830px;
  height: 40vh;
  min-height: 600px;
  /*background-color: red*/
}
</style>