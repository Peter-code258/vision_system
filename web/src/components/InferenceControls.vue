<template>
  <el-card>
    <h3>Inference Controls</h3>
    <el-switch v-model="isRunning" active-text="Running" inactive-text="Stopped" @change="toggleInference"/>
    <el-slider v-model="confidence" :min="0" :max="1" step="0.01" show-tooltip>
      <template #label>
        Confidence: {{ confidence.toFixed(2) }}
      </template>
    </el-slider>
  </el-card>
</template>

<script>
import axios from 'axios'
export default {
  data() {
    return {
      isRunning: false,
      confidence: 0.5
    }
  },
  methods: {
    toggleInference(val) {
      axios.post('/api/inference_toggle', { running: val })
    },
    updateConfidence() {
      axios.post('/api/set_confidence', { confidence: this.confidence })
    }
  },
  watch: {
    confidence: 'updateConfidence'
  }
}
</script>

<style scoped>
</style>
