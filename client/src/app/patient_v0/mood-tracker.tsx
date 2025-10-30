// "use client"

// import { useState, useEffect } from "react"
// import { Button } from "../../components/ui/button"
// import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../components/ui/card"
// import { Textarea } from "../../components/ui/textarea"

// const MOODS = [
//   { emoji: "😢", label: "Sad", value: "sad" },
//   { emoji: "😐", label: "Neutral", value: "neutral" },
//   { emoji: "🙂", label: "Happy", value: "happy" },
//   { emoji: "😄", label: "Excited", value: "excited" },
//   { emoji: "😍", label: "Loved", value: "loved" },
// ]

// export default function MoodTracker({ userId }: { userId: string }) {
//   const [selectedMood, setSelectedMood] = useState<string | null>(null)
//   const [intensity, setIntensity] = useState(5)
//   const [notes, setNotes] = useState("")
//   const [isLoading, setIsLoading] = useState(false)
//   const [moods, setMoods] = useState<any[]>([])
//   const [message, setMessage] = useState("")

//   useEffect(() => {
//     fetchMoods()
//   }, [userId])

//   const fetchMoods = async () => {
//     try {
//       const token = localStorage.getItem("token")
//       const res = await fetch(`/api/mood?userId=${userId}`, {
//         headers: { Authorization: `Bearer ${token}` },
//       })
//       if (res.ok) {
//         const data = await res.json()
//         setMoods(data)
//       }
//     } catch (error) {
//       console.error("Failed to fetch moods:", error)
//     }
//   }

//   const handleSubmit = async () => {
//     if (!selectedMood) {
//       setMessage("Please select a mood")
//       return
//     }

//     setIsLoading(true)
//     setMessage("")

//     try {
//       const token = localStorage.getItem("token")
//       const res = await fetch("/api/mood", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//           Authorization: `Bearer ${token}`,
//         },
//         body: JSON.stringify({
//           userId,
//           mood: selectedMood,
//           intensity,
//           notes,
//         }),
//       })

//       if (res.ok) {
//         setMessage("Mood logged successfully!")
//         setSelectedMood(null)
//         setIntensity(5)
//         setNotes("")
//         fetchMoods()
//       } else {
//         setMessage("Failed to log mood")
//       }
//     } catch (error) {
//       setMessage("An error occurred")
//     } finally {
//       setIsLoading(false)
//     }
//   }

//   return (
//     <div className="space-y-6">
//       <Card>
//         <CardHeader>
//           <CardTitle>How are you feeling today?</CardTitle>
//           <CardDescription>Track your mood to help us understand your mental health journey</CardDescription>
//         </CardHeader>
//         <CardContent className="space-y-6">
//           <div className="flex justify-center gap-4">
//             {MOODS.map((mood) => (
//               <button
//                 key={mood.value}
//                 onClick={() => setSelectedMood(mood.value)}
//                 className={`flex flex-col items-center gap-2 p-4 rounded-lg transition-all ${
//                   selectedMood === mood.value
//                     ? "bg-primary text-primary-foreground scale-110"
//                     : "bg-muted hover:bg-muted/80"
//                 }`}
//               >
//                 <span className="text-4xl">{mood.emoji}</span>
//                 <span className="text-sm font-medium">{mood.label}</span>
//               </button>
//             ))}
//           </div>

//           <div className="space-y-2">
//             <label className="text-sm font-medium">Intensity (1-10)</label>
//             <input
//               type="range"
//               min="1"
//               max="10"
//               value={intensity}
//               onChange={(e) => setIntensity(Number(e.target.value))}
//               className="w-full"
//             />
//             <div className="text-center text-sm text-muted-foreground">{intensity}/10</div>
//           </div>

//           <div className="space-y-2">
//             <label className="text-sm font-medium">Notes (optional)</label>
//             <Textarea
//               placeholder="What's on your mind?"
//               value={notes}
//               onChange={(e) => setNotes(e.target.value)}
//               className="min-h-24"
//             />
//           </div>

//           {message && (
//             <div
//               className={`p-3 rounded-md text-sm ${
//                 message.includes("successfully") ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"
//               }`}
//             >
//               {message}
//             </div>
//           )}

//           <Button onClick={handleSubmit} disabled={isLoading} className="w-full">
//             {isLoading ? "Logging..." : "Log Mood"}
//           </Button>
//         </CardContent>
//       </Card>

//       {moods.length > 0 && (
//         <Card>
//           <CardHeader>
//             <CardTitle>Recent Moods</CardTitle>
//           </CardHeader>
//           <CardContent>
//             <div className="space-y-2">
//               {moods.slice(0, 5).map((mood: any) => (
//                 <div key={mood._id} className="flex items-center justify-between p-3 bg-muted rounded-lg">
//                   <div className="flex items-center gap-3">
//                     <span className="text-2xl">{MOODS.find((m) => m.value === mood.mood)?.emoji}</span>
//                     <div>
//                       <p className="font-medium capitalize">{mood.mood}</p>
//                       <p className="text-sm text-muted-foreground">{new Date(mood.createdAt).toLocaleDateString()}</p>
//                     </div>
//                   </div>
//                   <div className="text-right">
//                     <p className="font-medium">{mood.intensity}/10</p>
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
