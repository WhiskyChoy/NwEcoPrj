<template lang="pug">
  div#app
    h1.title Dynamic Aviation Network Under Pandemic
    h2.subtitle A Demo For Course CIE6036
    div.section
      div.section_title INTRODUCTION
      div.section_content(ref="introduction") Inspired by the fusing command from civil aviation, the decision making of dynamic airline system takes a more essential role in current situation. After we propose a new decision making model on airline capacity and solve a constrained optimization problem, we obtain empirical experiment results. To better illustrate the relationship between decision making with different #[strong dates/cities/$\epsilon$/$T$], we now apply echarts to our Dynamic Airline System in 3D.
    GlobeChart
    div.section
      div.section_title ILLUSTRATION
      div.section_content(ref="illustration") To further illustrate, we explain several characteristics: <br> 1. Relative Capacity: $c_{ij}^* / c_{ij}^{max}$ (optimal airline load rate) <br> 2. Absolute Capacity: $c_{ij}^*$ (optimal airline capacity) <br> 3. Date: $S_i$ (infection severity of different cities) <br> 4. Relaxtion parameter: $(\epsilon,T)$
    div.section
      div.section_title CONCLUSION
      div.section_content In sum, we could draw several meaningful conclusions: <br> 1. Reflect the tradeoff between the satisfaction of passengers’ transportation demand and the avoidance of disease spreading <br> 2. Show the necessity to greatly reduce the capacity of airline from the cities with severe outbreaks ("Five One" policy)
    GlobeTotalChart.global-total-chart
    div.section
      div.section_title FUTURE WORK
      div.section_content(ref="future_work") 1. Decide the flight to/between high risk area better <br> 2. Generate the $c^{max}$ matrix by the whole year's data <br> 3. Get the data of other main countries like U.S. and U.K
    div.section
      div.section_title REFERENCES
      div.section_content.references [1] The impacts of covid-19 on the global airline industry: An event study approach. Journal of air transport man agement, 2020. <br> [2] Scenarios for a post-covid-19 world airline network. arXiv preprint 2020. <br> [3] Temporal characteristics of the chinese aviation network and their effects on the spread of infectious diseases. Scientific Reports, 2017. <br> [4] European air lines' strategic responses to the covid-19 pandemic (January-May, 2020). Journal of air transport man agement, 2020. <br> [5] European airline response to the covid-19 pandemic–contraction, consolidation and future considerations for airline business and management. Research in Transportation Business & Management, 2020. <br> [6] Geatpy: the genetic and evolutionary algorithm toolbox with high performance in python, 2020. <br> [7] Air quality forecast monitoring and its impact on brain health based on big data and the internet of things. IEEE Access, 2018. <br> [8] A sidarthe model of covid-19 epidemic in Italy. arXiv preprint, 2020.<br> [9] Extended sir prediction of the epidemics trend of covid-19 in Italy and compared with hunan, China. Frontiers in medicine, 2020. <br> [10] Prevention and control of covid-19 in public transportation: experience from China. Environmental pollution, 2020. <br> [11] Early dynamics of transmission and control of covid-19: a mathematical modelling study. The lancet infectious diseases, 2020.
</template>

<script>

import GlobeChart from "@/components/GlobeChart";
import GlobeTotalChart from "@/components/GlobeTotalChart";
export default {
  name: 'App',
  components: {
    GlobeTotalChart,
    GlobeChart
  },
  methods:{
    getDoms(...refs){
      let result = [];
      for(let ref of refs){
        result.push(this.$refs[ref])
      }
      return result
    }
  },
  async mounted() {
    await this.$nextTick()
    if(!this.$mathjax.isMathjaxConfig){//判断是否初始配置，若无则配置。
      this.$mathjax.initMathjaxConfig();
    }
    this.$mathjax.MathQueue(this.getDoms('introduction','illustration','future_work'));//传入组件id，让组件被MathJax渲染
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 1rem;
  display: flex;
  flex-direction: column;
  align-items: center;
}

body{
  margin: 0
}

/* MathJax v2.7.5 from 'cdnjs.cloudflare.com' */
.mjx-chtml {
  outline: 0;
}
.MJXc-display {
  overflow-x: auto;
  overflow-y: hidden;
}

.global-total-chart{
  margin-top: 1rem;
}

.title{
  width: 100%;
  background-color: white;
  position: sticky;
  top: 0;
  z-index: 100;
}

.subtitle{
  margin-top: 0;
}

.section{
  max-width: 1000px;
  width: 70%;
  min-width: 850px;
  margin-bottom: 15px;
  text-align: left;
  margin-top: 1rem;
}

.section_title{
  font-weight: bolder;
  font-size: x-large;
  border-bottom: black thin solid;
}

.section_content{
  font-size: large;
  line-height: 150%;
  text-align: justify;
  margin-top: 0.5rem;
}

.references{
  color: rgb(100,100,100);
}

</style>
