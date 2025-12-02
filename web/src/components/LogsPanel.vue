<template>
  <el-card>
    <h3>Logs</h3>
    <div class="logs">
      <div v-for="(line, index) in logs" :key="index">{{ line }}</div>
    </div>
    <el-button type="primary" @click="fetchLogs" style="margin-top:10px;">Refresh</el-button>
  </el-card>
</template>

<script>
import axios from 'axios'
export default {
  data() {
    return {
      logs: []
    }
  },
  methods: {
    fetchLogs() {
      axios.get('/api/logs').then(res => {
        this.logs = res.data
      })
    }
  },
  mounted() {
    this.fetchLogs()
  }
}
</script>

<style scoped>
.logs {
  max-height: 200px;
  overflow-y: auto;
  font-family: monospace;
  background-color: #f9f9f9;
  padding: 5px;
  border: 1px solid #ccc;
}
</style>
