"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../components/ui/card"
import { Button } from "../../components/ui/button"
import { Input } from "../../components/ui/input"

export default function PatientList({
  therapistId,
  onSelectPatient,
}: {
  therapistId: string
  onSelectPatient: (patientId: string) => void
}) {
  const [patients, setPatients] = useState<any[]>([])
  const [filteredPatients, setFilteredPatients] = useState<any[]>([])
  const [searchTerm, setSearchTerm] = useState("")
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    fetchPatients()
  }, [therapistId])

  useEffect(() => {
    const filtered = patients.filter(
      (p) =>
        p.firstName.toLowerCase().includes(searchTerm.toLowerCase()) ||
        p.lastName.toLowerCase().includes(searchTerm.toLowerCase()) ||
        p.email.toLowerCase().includes(searchTerm.toLowerCase()),
    )
    setFilteredPatients(filtered)
  }, [searchTerm, patients])

  const fetchPatients = async () => {
    try {
      const token = localStorage.getItem("token")
      const res = await fetch(`/api/therapist/patients?therapistId=${therapistId}`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (res.ok) {
        const data = await res.json()
        setPatients(data)
      }
    } catch (error) {
      console.error("Failed to fetch patients:", error)
    } finally {
      setIsLoading(false)
    }
  }

  if (isLoading) {
    return <div className="text-center text-muted-foreground">Loading patients...</div>
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Your Patients</CardTitle>
        <CardDescription>Manage and monitor your patient list</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Input
          placeholder="Search patients..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="mb-4"
        />

        <div className="space-y-2">
          {filteredPatients.length > 0 ? (
            filteredPatients.map((patient) => (
              <div
                key={patient._id}
                className="flex items-center justify-between p-4 bg-muted rounded-lg hover:bg-muted/80"
              >
                <div>
                  <p className="font-semibold">
                    {patient.firstName} {patient.lastName}
                  </p>
                  <p className="text-sm text-muted-foreground">{patient.email}</p>
                </div>
                <Button size="sm" onClick={() => onSelectPatient(patient._id)}>
                  View
                </Button>
              </div>
            ))
          ) : (
            <p className="text-center text-muted-foreground py-8">No patients found</p>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
