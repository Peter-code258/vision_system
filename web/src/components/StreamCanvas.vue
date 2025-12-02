<template>
  <el-card style="flex:1">
    <h3>Live Stream</h3>
    <canvas ref="canvas" width="640" height="480"></canvas>
  </el-card>
</template>

<script>
export default {
  data() {
    return {
      ws: null
    }
  },
  mounted() {
    this.ws = new WebSocket('ws://localhost:8000/ws/stream')
    this.ws.onmessage = event => {
      const img = new Image()
      img.src = 'data:image/jpeg;base64,' + event.data
      img.onload = () => {
        const canvas = this.$refs.canvas
        const ctx = canvas.getContext('2d')
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
      }
    }
  },
  beforeUnmount() {
    if (this.ws) this.ws.close()
  }
}
</script>

<style scoped>
canvas {
  width: 100%;
  height: auto;
  border: 1px solid #ccc;
}
</style>
