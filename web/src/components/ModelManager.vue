<template>
  <el-card>
    <h3>Model Manager</h3>
    <el-upload
      action="/api/upload_model"
      :show-file-list="true"
      :on-success="onUploadSuccess">
      <el-button>Upload Model</el-button>
    </el-upload>
    <el-select v-model="selectedModel" placeholder="Select model" style="margin-top:10px;width:100%">
      <el-option
        v-for="model in models"
        :key="model"
        :label="model"
        :value="model">
      </el-option>
    </el-select>
    <el-button type="primary" style="margin-top:10px;" @click="loadModel">Load Model</el-button>
  </el-card>
</template>

<script>
import axios from 'axios'
export default {
  data() {
    return {
      models: [],
      selectedModel: null
    }
  },
  methods: {
    fetchModels() {
      axios.get('/api/models').then(res => {
        this.models = res.data
      })
    },
    loadModel() {
      if (!this.selectedModel) return
      axios.post('/api/load_model', { model: this.selectedModel })
        .then(() => this.$message.success('Model loaded'))
    },
    onUploadSuccess() {
      this.fetchModels()
    }
  },
  mounted() {
    this.fetchModels()
  }
}
</script>

<style scoped>
</style>
