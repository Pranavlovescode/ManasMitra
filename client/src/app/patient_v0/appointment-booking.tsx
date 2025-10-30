// "use client"

// import { useState, useEffect } from "react"
// import { Button } from "../../components/ui/button"
// import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../components/ui/card"
// import { Input } from "../../components/ui/input"

// export default function AppointmentBooking({ userId }: { userId: string }) {
//   const [appointments, setAppointments] = useState<any[]>([])
//   const [showBooking, setShowBooking] = useState(false)
//   const [selectedDate, setSelectedDate] = useState("")
//   const [selectedTime, setSelectedTime] = useState("")
//   const [sessionType, setSessionType] = useState<"individual" | "group">("individual")
//   const [isLoading, setIsLoading] = useState(false)

//   useEffect(() => {
//     fetchAppointments()
//   }, [userId])

//   const fetchAppointments = async () => {
//     try {
//       const token = localStorage.getItem("token")
//       const res = await fetch(`/api/appointments?userId=${userId}`, {
//         headers: { Authorization: `Bearer ${token}` },
//       })
//       if (res.ok) {
//         const data = await res.json()
//         setAppointments(data)
//       }
//     } catch (error) {
//       console.error("Failed to fetch appointments:", error)
//     }
//   }

//   const handleBookAppointment = async () => {
//     if (!selectedDate || !selectedTime) {
//       alert("Please select date and time")
//       return
//     }

//     setIsLoading(true)

//     try {
//       const token = localStorage.getItem("token")
//       const res = await fetch("/api/appointments", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//           Authorization: `Bearer ${token}`,
//         },
//         body: JSON.stringify({
//           patientId: userId,
//           scheduledAt: new Date(`${selectedDate}T${selectedTime}`),
//           type: sessionType,
//         }),
//       })

//       if (res.ok) {
//         setShowBooking(false)
//         setSelectedDate("")
//         setSelectedTime("")
//         fetchAppointments()
//       }
//     } catch (error) {
//       console.error("Failed to book appointment:", error)
//     } finally {
//       setIsLoading(false)
//     }
//   }

//   return (
//     <div className="space-y-6">
//       {!showBooking ? (
//         <Card>
//           <CardHeader>
//             <CardTitle>Book an Appointment</CardTitle>
//             <CardDescription>Schedule a therapy session with your therapist</CardDescription>
//           </CardHeader>
//           <CardContent>
//             <Button onClick={() => setShowBooking(true)} className="w-full">
//               Book New Appointment
//             </Button>
//           </CardContent>
//         </Card>
//       ) : (
//         <Card>
//           <CardHeader>
//             <CardTitle>Schedule Your Session</CardTitle>
//           </CardHeader>
//           <CardContent className="space-y-4">
//             <div className="space-y-2">
//               <label className="text-sm font-medium">Session Type</label>
//               <div className="flex gap-4">
//                 <label className="flex items-center gap-2 cursor-pointer">
//                   <input
//                     type="radio"
//                     value="individual"
//                     checked={sessionType === "individual"}
//                     onChange={(e) => setSessionType(e.target.value as "individual" | "group")}
//                   />
//                   <span className="text-sm">Individual</span>
//                 </label>
//                 <label className="flex items-center gap-2 cursor-pointer">
//                   <input
//                     type="radio"
//                     value="group"
//                     checked={sessionType === "group"}
//                     onChange={(e) => setSessionType(e.target.value as "individual" | "group")}
//                   />
//                   <span className="text-sm">Group</span>
//                 </label>
//               </div>
//             </div>

//             <div className="space-y-2">
//               <label className="text-sm font-medium">Date</label>
//               <Input type="date" value={selectedDate} onChange={(e) => setSelectedDate(e.target.value)} />
//             </div>

//             <div className="space-y-2">
//               <label className="text-sm font-medium">Time</label>
//               <Input type="time" value={selectedTime} onChange={(e) => setSelectedTime(e.target.value)} />
//             </div>

//             <div className="flex gap-2">
//               <Button onClick={() => setShowBooking(false)} variant="outline" className="flex-1">
//                 Cancel
//               </Button>
//               <Button onClick={handleBookAppointment} disabled={isLoading} className="flex-1">
//                 {isLoading ? "Booking..." : "Book Appointment"}
//               </Button>
//             </div>
//           </CardContent>
//         </Card>
//       )}

//       {appointments.length > 0 && (
//         <Card>
//           <CardHeader>
//             <CardTitle>Your Appointments</CardTitle>
//           </CardHeader>
//           <CardContent>
//             <div className="space-y-3">
//               {appointments.map((apt: any) => (
//                 <div key={apt._id} className="p-4 bg-muted rounded-lg">
//                   <div className="flex items-center justify-between">
//                     <div>
//                       <p className="font-semibold capitalize">{apt.type} Session</p>
//                       <p className="text-sm text-muted-foreground">{new Date(apt.scheduledAt).toLocaleString()}</p>
//                     </div>
//                     <span
//                       className={`px-3 py-1 rounded-full text-xs font-medium ${
//                         apt.status === "scheduled"
//                           ? "bg-blue-100 text-blue-800"
//                           : apt.status === "completed"
//                             ? "bg-green-100 text-green-800"
//                             : "bg-red-100 text-red-800"
//                       }`}
//                     >
//                       {apt.status}
//                     </span>
//                   </div>
//                 </div>
//               ))}
//             </div>
//           </CardContent>
//         </Card>
//       )}
//     </div>
//   )
// }
