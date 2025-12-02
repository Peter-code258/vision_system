<template>
  <el-card>
    <h3>Heatmap Settings</h3>
    <el-slider v-model="threshold" :min="0" :max="255" step="1" show-tooltip>
      <template #label>
        Threshold: {{ threshold }}
      </template>
    </el-slider>
    <el-button type="primary" @click="applySettings" style="margin-top:10px;">Apply</el-button>
  </el-card>
</template>

<script>
import axios from 'axios'
export default {
  data() {
    return {
      threshold: 128
    }
  },
  methods: {
    applySettings() {
      axios.post('/api/set_heatmap', { threshold: this.threshold })
    }
  }
}
</script>

<style scoped>
</style>
