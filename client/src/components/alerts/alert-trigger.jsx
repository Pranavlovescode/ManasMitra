"use client"

import { useEffect } from "react"

export default function AlertTrigger({ userId, userRole }) { // Type annotation removed
  useEffect(() => {
    checkAndTriggerAlerts()
    const interval = setInterval(checkAndTriggerAlerts, 300000) // Check every 5 minutes
    return () => clearInterval(interval)
  }, [userId, userRole])

  const checkAndTriggerAlerts = async () => {
    try {
      const token = localStorage.getItem("token")
      const res = await fetch("/api/alerts/check", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ userId, userRole }),
      })

      if (res.ok) {
        const data = await res.json()
        if (data.alerts && data.alerts.length > 0) {
          // Trigger notifications
          data.alerts.forEach((alert) => { // Type :any removed
            sendNotification(alert)
          })
        }
      }
    } catch (error) {
      console.error("Failed to check alerts:", error)
    }
  }

  const sendNotification = async (alert) => { // Type :any removed
    try {
      const token = localStorage.getItem("token")
      await fetch("/api/notifications/send", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          userId,
          title: alert.title,
          message: alert.message,
          type: alert.severity === "high" ? "error" : "warning",
        }),
      })
    } catch (error) {
      console.error("Failed to send notification:", error)
    }
  }

  return null
}